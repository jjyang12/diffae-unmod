import argparse
import os
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn, distributed, amp
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import AUROC
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import MNIST_dataset
from choices import TrainMode

from experiment import is_time, LitModel, train
from model import BeatGANsAutoencModel
from renderer import render_uncondition
from templates import mnist_autoenc

np.random.seed(0)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class OneClassModel(LightningModule):
    def __init__(self, encoder, l: float, batch_size: int, lr: float):
        super().__init__()
        self.l = l
        self.batch_size = batch_size
        self.lr = lr
        self.encoder = encoder
        self.validation_step_outputs = []
        self.validation_step_labels = []

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.encoder.conf.seed is not None:
            seed = self.encoder.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################
        self.train_data = MNIST_dataset(path='/home/eprakash/mnist/train_list.txt')
        self.val_data = MNIST_dataset(path='/home/eprakash/mnist/test_list.txt')
        print('train data:', len(self.train_data))
        print('val data:', len(self.val_data))

        self.load_semantic_average()

    def train_dataloader(self):
        print('on train dataloader start ...')
        sampler = DistributedSampler(self.train_data, shuffle=True, drop_last=True)
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler, shuffle=False,
                          num_workers=2,
                          pin_memory=True,
                          drop_last=True,
                          multiprocessing_context=get_context('fork'),)

    def val_dataloader(self):
        print('on val dataloader start ...')
        sampler = DistributedSampler(self.val_data, shuffle=False, drop_last=False)
        return DataLoader(self.val_data, batch_size=min(len(self.val_data) // get_world_size(), self.batch_size),
                          sampler=sampler,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True,
                          drop_last=True,
                          multiprocessing_context=get_context('fork'),)

    # def test_dataloader(self):
    #     print('on test dataloader start ...')
    #     print(self.batch_size)
    #     sampler = DistributedSampler(self.test_data, shuffle=False, drop_last=False)
    #     return DataLoader(self.test_data, batch_size=min(len(self.test_data) // get_world_size(), self.batch_size),
    #                       sampler=sampler,
    #                       shuffle=False,
    #                       num_workers=2,
    #                       pin_memory=True,
    #                       drop_last=True,
    #                       multiprocessing_context=get_context('fork'), )

    def training_step(self, batch, batch_idx):
        x = batch['img']

        features = self.encoder.model.encoder.forward(x)
        dists = torch.sum((features - self.c.to(self.device)) ** 2, dim=1)
        oneclass_loss = torch.mean(dists)

        # sample t's from a uniform distribution
        t, weight = self.encoder.T_sampler.sample(len(x), x.device)
        losses = self.encoder.sampler.training_losses(model=self.encoder.model,
                                              x_start=x,
                                              t=t)
        diffae_loss = losses['loss'].mean()
        # divide by accum batches to make the accumulated gradient exact!
        for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
            if key in losses:
                losses[key] = self.all_gather(losses[key]).mean()

        if self.global_rank == 0:
            self.log('loss', losses['loss'])
            for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    self.logger.experiment.add_scalar(
                        f'loss/{key}', losses[key], self.encoder.num_samples)

        loss = diffae_loss + self.l * oneclass_loss

        self.log("total_loss", loss, prog_bar=True)
        self.log("oneclass_loss", oneclass_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']

        features = self.encoder.encode(x)
        dists = torch.sum((features - self.c.to(self.device)) ** 2, dim=1)

        self.validation_step_outputs.append(dists)
        self.validation_step_labels.append(y)

    def on_validation_epoch_end(self) -> None:
        all_preds = torch.stack(self.validation_step_outputs)
        norm_preds = all_preds / torch.max(all_preds)
        all_labels = torch.stack(self.validation_step_labels)
        auroc = AUROC(task='binary')(norm_preds, all_labels)
        self.log("auroc", auroc)
        print(auroc)
        self.validation_step_outputs.clear()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.encoder.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.encoder.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.encoder.model.latent_net, self.encoder.ema_model.latent_net,
                    self.encoder.conf.ema_decay)
            else:
                ema(self.encoder.model, self.encoder.ema_model, self.encoder.conf.ema_decay)

            # logging
            if self.encoder.conf.train_mode.require_dataset_infer():
                imgs = None
            else:
                imgs = batch['img']
            self.log_sample(x_start=imgs)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer,
                                 optimizer_idx: int) -> None:
        self.encoder.on_before_optimizer_step(optimizer, optimizer_idx)

    def log_sample(self, x_start):
        """
        put images to the tensorboard
        """
        def do(model,
               postfix,
               use_xstart,
               save_real=False,
               no_latent_diff=False,
               interpolate=False):
            model.eval()
            with torch.no_grad():
                all_x_T = self.encoder.split_tensor(self.encoder.x_T)
                batch_size = min(len(all_x_T), self.encoder.conf.batch_size_eval)
                # allow for superlarge models
                loader = DataLoader(all_x_T, batch_size=batch_size)

                Gen = []
                for x_T in loader:
                    if use_xstart:
                        _xstart = x_start[:len(x_T)]
                    else:
                        _xstart = None

                    if self.encoder.conf.train_mode.is_latent_diffusion(
                    ) and not use_xstart:
                        # diffusion of the latent first
                        gen = render_uncondition(
                            conf=self.encoder.conf,
                            model=model,
                            x_T=x_T,
                            sampler=self.encoder.eval_sampler,
                            latent_sampler=self.encoder.eval_latent_sampler,
                            conds_mean=self.encoder.conds_mean,
                            conds_std=self.encoder.conds_std)
                    else:
                        if not use_xstart and self.encoder.conf.model_type.has_noise_to_cond(
                        ):
                            model: BeatGANsAutoencModel
                            # special case, it may not be stochastic, yet can sample
                            cond = torch.randn(len(x_T),
                                               self.encoder.conf.style_ch,
                                               device=self.device)
                            cond = model.noise_to_cond(cond)
                        else:
                            if interpolate:
                                with amp.autocast(self.encoder.conf.fp16):
                                    cond = model.encoder(_xstart)
                                    i = torch.randperm(len(cond))
                                    cond = (cond + cond[i]) / 2
                            else:
                                cond = None
                        gen = self.encoder.eval_sampler.sample(model=model,
                                                       noise=x_T,
                                                       cond=cond,
                                                       x_start=_xstart)
                    Gen.append(gen)

                gen = torch.cat(Gen)
                gen = self.all_gather(gen)
                if gen.dim() == 5:
                    # (n, c, h, w)
                    gen = gen.flatten(0, 1)

                if save_real and use_xstart:
                    # save the original images to the tensorboard
                    real = self.all_gather(_xstart)
                    if real.dim() == 5:
                        real = real.flatten(0, 1)

                    if self.global_rank == 0:
                        vid_real = (make_grid(real + 1) / 2) * 256
                        self.logger.experiment.log({
                            f'sample{postfix}/real': wandb.Image(vid_real.cpu())
                        })

                if self.global_rank == 0:
                    # save samples to wandb
                    vid = ((gen + 1) / 2) * 256
                    # sample_dir = os.path.join(self.encoder.conf.logdir,
                    #                           f'sample{postfix}')
                    # if not os.path.exists(sample_dir):
                    #     os.makedirs(sample_dir)
                    # path = os.path.join(sample_dir,
                    #                     '%d.png' % self.num_samples)
                    # save_image(grid, path)
                    self.logger.experiment.log({
                        f'sample{postfix}': wandb.Image(vid.cpu())
                    })
            model.train()

        if self.encoder.conf.sample_every_samples > 0 and is_time(
                self.encoder.num_samples, self.encoder.conf.sample_every_samples,
                self.encoder.conf.batch_size_effective):

            if self.encoder.conf.train_mode.require_dataset_infer():
                do(self.encoder.model, '', use_xstart=False)
                do(self.encoder.ema_model, '_ema', use_xstart=False)
            else:
                if self.encoder.conf.model_type.has_autoenc(
                ) and self.encoder.conf.model_type.can_sample():
                    do(self.encoder.model, '', use_xstart=False)
                    do(self.encoder.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.encoder.model, '_enc', use_xstart=True, save_real=True)
                    do(self.encoder.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                elif self.encoder.conf.train_mode.use_latent_net():
                    do(self.encoder.model, '', use_xstart=False)
                    do(self.encoder.ema_model, '_ema', use_xstart=False)
                    # autoencoding mode
                    do(self.encoder.model, '_enc', use_xstart=True, save_real=True)
                    do(self.encoder.model,
                       '_enc_nodiff',
                       use_xstart=True,
                       save_real=True,
                       no_latent_diff=True)
                    do(self.encoder.ema_model,
                       '_enc_ema',
                       use_xstart=True,
                       save_real=True)
                else:
                    do(self.encoder.model, '', use_xstart=True, save_real=True)
                    do(self.encoder.ema_model, '_ema', use_xstart=True, save_real=True)

    def configure_optimizers(self):
        solver = torch.optim.Adam(nn.ModuleList([self.encoder]).parameters(), lr=self.lr)
        return solver

    def load_semantic_average(self):
        path = Path(f"checkpoints/{args.pretrain_dir}/latent.pkl")
        if path.exists():
            self.c = torch.load(path)['conds_mean']
            # assert self.c.shape == (512,), f"{self.c.shape} =/= (512,)"
            return
        raise FileNotFoundError(path)


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def main(args):

    if args.cv:
        # do 5 fold cross validation
        base_dir = [args.dir + "/" + str(i) for i in range(5)]
        cv_fold = range(5)
    else:
        base_dir = [args.dir]
        cv_fold = [None]


    gpus = [0, 1, 2, 3]
    nodes = 1
    if not Path(f"checkpoints/{args.pretrain_dir}/latent.pkl").exists():
        conf = mnist_autoenc()
        conf.base_dir = f'checkpoints'  # "checkpoints_jh2"
        conf.name = f'{args.pretrain_dir}'  # "checkpoints_jh2"
        conf.eval_programs = ['infer']
        train(conf, gpus=gpus, mode='eval')

    for dir, cv in zip(base_dir, cv_fold):
        np.random.seed(8)
        conf = mnist_autoenc()
        # from choices import TrainMode
        # conf.train_mode = TrainMode.diffusion
        conf.base_dir = dir  # "checkpoints_jh2"
        # conf.pretrain.path = 'diffae/checkpoints/ffhq128_autoenc/last.ckpt'
        # conf.latent_infer_path = 'diffae/checkpoints/ffhq128_autoenc/latent.pkl'
        model = LitModel(conf)
        state = torch.load(f'checkpoints/{args.pretrain_dir}/last.ckpt', map_location='cpu')
        print(model.load_state_dict(state['state_dict'], strict=False))

        gan = OneClassModel(
            model,
            args.l,
            args.batch_size,
            args.lr
        )

        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                     save_last=True,
                                     save_top_k=1,
                                     every_n_train_steps=conf.save_every_samples //
                                     conf.batch_size_effective)
        checkpoint_path = f'{conf.logdir}/last.ckpt'
        print('ckpt path:', checkpoint_path)
        if os.path.exists(checkpoint_path):
            resume = checkpoint_path
            print('resume!')
        else:
            if conf.continue_from is not None:
                # continue from a checkpoint
                resume = conf.continue_from.path
            else:
                resume = None

        wandb_logger = WandbLogger(
            project="oneclass",
            entity="vid-anomaly-detect",
            config=args,
            save_dir=conf.logdir)

        if len(gpus) == 1 and nodes == 1:
            strategy = None
        else:
            strategy = DDPStrategy(find_unused_parameters=False)

        print("training now")
        trainer = Trainer(
            max_epochs=5000,  # 1000 epochs is usually sufficient for v-space
            resume_from_checkpoint=resume,
            gpus=gpus,
            num_nodes=nodes,
            accelerator="gpu",
            strategy=strategy,
            precision=16 if conf.fp16 else 32,
            callbacks=[
                checkpoint,
                LearningRateMonitor(),
            ],
            # clip in the model instead
            # gradient_clip_val=conf.grad_clip,
            replace_sampler_ddp=True,
            logger=wandb_logger,
            accumulate_grad_batches=conf.accum_batches,
        )

        trainer.fit(gan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--l",
        required=True,
        type=float,
        help="CI regularization strength"
    )
    parser.add_argument(
        "--lr",
        required=True,
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        default=32,
        help="batch size"
    )

    parser.add_argument(
        "--pretrain_dir",
        required=True,
        help="model input directory"
    )

    parser.add_argument(
        "--dir",
        default="run_" + datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        help="model save directory"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="perform 5-fold cross validation"
    )

    args = parser.parse_args()

    main(args)
