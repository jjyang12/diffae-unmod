import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm
import sys

from dataset import MNIST_dataset
from experiment import LitModel
from templates import mnist_autoenc

from metrics import lpips
from ssim import ssim
from metrics import psnr

torch.cuda.empty_cache()

NUM_GPUS = 4 # Number of total GPUs being used, should be set to 1 if you are running python eval_st_anomaly.py with no args
batch_size = 4 # 64 is fastest?
cuda_device = 0
if len(sys.argv) > 1:
    cuda_device = int(sys.argv[1])
else:
    NUM_GPUS = 1
print(NUM_GPUS)
device = 'cuda:' + str(cuda_device)

conf = mnist_autoenc()
print(conf.name)
conf.net_ch = 32
model = LitModel(conf)
state = torch.load(f'checkpoints/mnist_2/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)
model.to(device)


data = MNIST_dataset(path="/home/eprakash/mnist/test_list.txt", image_size=conf.img_size, cf_stride=False)
num_batches = int(len(data)/batch_size) + 1
start_batch = (cuda_device * num_batches) // NUM_GPUS # Floored lower bound
end_batch = ((cuda_device + 1) * num_batches) // NUM_GPUS # Floored upper bound
print(str(cuda_device) + ": processing {} batches from idx {} to idx {}".format(num_batches, start_batch, end_batch))
avg_loss = []

with torch.no_grad():
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    for b in tqdm(range(start_batch, end_batch)):
        #Building batch
        batch_len = batch_size
        if (b == num_batches - 1):
            batch_len = len(data) - batch_size * b
            if batch_len == 0:
                break
        frame_batch = torch.zeros(size=(batch_len, 3, conf.img_size, conf.img_size), device=device)
        frame_batch_ori = torch.zeros(size=(batch_len, 3, conf.img_size, conf.img_size), device=device)
        for i in range(batch_len):
            img_i = data[batch_size * b + i]['img']
            frame_batch[i] = img_i
            frame_batch_ori[i] = data[batch_size * b + i]['orig_img']
        
        #Encoding
        model.model.to(device)
        cond = model.encode(frame_batch.to(device))
        xT = model.encode_stochastic(frame_batch, cond, T=None)
        xT_rand = torch.rand(xT.shape).to(device)
        #Decoding
        pred = model.render(xT_rand, cond, T=None)
        ori = frame_batch_ori

        #Calculating new metrics
        for j in range(batch_len):
            ssim_scores = []
            psnr_scores = []
            lpips_scores = []
            mse_scores = []
            ssim_scores.append(ssim(ori[j].reshape((1,3,64,64)), pred[j].reshape((1,3,64,64)), size_average=False).item())
            psnr_scores.append(psnr(ori[j].reshape((1,3,64,64)), pred[j].reshape((1,3,64,64))).item())
            lpips_scores.append(lpips_fn.forward(ori[j], pred[j]).view(-1).item())
            mse_scores.append((ori[j] - pred[j]).pow(2).mean(dim=[0,1,2]).item())
            
            methods = {"ssim": np.mean(ssim_scores), "psnr": np.mean(psnr_scores), "lpips": np.mean(lpips_scores), "mse": np.mean(mse_scores)}
            for m in methods:
                with open("mnist_noflips_32actual_noema_fixeddl_randsto_"+ m + "_obj_{}.log".format(cuda_device), "a") as fp:
                    if (j == 0):
                        fp.write("Batch results: ")
                    fp.write(str(methods[m]) + "|")
                    if (j == (batch_len - 1)):
                        fp.write("\n")

print("DONE!")