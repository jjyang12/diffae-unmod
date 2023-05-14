from templates import *
from dataset import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader

device = 'cuda:0'
conf = ubnormal_autoenc()
print(conf.name)
model = LitModel(conf)

state = torch.load(f'/home/jy2k16/diffae-unmod/checkpoints/ubnormal128_img_autoenc_1/last.ckpt', map_location=device)
#state = torch.load(f'checkpoints/video_autoenc/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.to(device)

data = UBNormal_dataset_eval(path="/home/jy2k16/UBNormalVids/test_list.txt", image_size=conf.img_size, label = "abnormal")
loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )

i = 0
for batch in loader:
    i += 1
    if i % 400 != 0 or i< 5000:
        continue
    if i > 7000:
        break

    print(f'starting example {i}')
    
    img = batch['img']
    orig_img = batch['orig_img']
    #img = img.permute((2,3,0,1))

    print(img.shape)
    print(orig_img.shape)

    print("Encoding...")
    cond = model.encode(img.to(device))
    #cond = torch.randn(1, 512, device=device)
    print(cond.shape)
    xT = model.encode_stochastic(x=img.to(device), cond=cond, T=250)
    #xT = torch.randn(1, 3, conf.img_size, conf.img_size, device=device)
    print(xT.shape)

    print("Decoding...")
    #cond = {'cond': cond}
    pred = model.render(noise=xT, cond=cond, T=20)

    print("Plotting...")
    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 5))

    ax[0].imshow(orig_img[0].permute(1, 2, 0).cpu())
    ax[1].imshow(pred[0].permute((1, 2, 0)).cpu())
    #img_name = "normal_batch/img_orig_" + str(i) + ".png"
    #save_image(orig_img[0].permute(1, 2, 0).cpu(), img_name)
    #img_name = "normal_batch/img_pred_" + str(i) + ".png"
    #save_image(pred.cpu(), img_name)
    plt.savefig("abnormal_visuals/batch_"+str(i)+".png")

    print("DONE!")


