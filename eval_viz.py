from templates import *
from dataset import *
from experiment import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch
import cv2

DATASET = "/home/eprakash/UCSD_Anomaly_Dataset.v1p2/UCSDped2/test_list.txt"
IDX = 1549

device = 'cuda:0'
conf = mnist_autoenc()
conf.net_ch = 32
model = LitModel(conf)


state = torch.load(f'/home/eprakash/diffae-unmod/checkpoints/ped2/last.ckpt', map_location=device)
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)

data = MNIST_dataset(path=DATASET, image_size=conf.img_size, cf_stride=False)
batch_len = 9
batch_size = 4
b = 0
frame_batch = torch.zeros(size=(batch_len, 3, conf.img_size, conf.img_size), device=device)
frame_batch_ori = torch.zeros(size=(batch_len, 3, conf.img_size, conf.img_size), device=device)
labels = []

labelpath = open("/home/eprakash/UCSD_Anomaly_Dataset.v1p2/UCSDped2/test_labels.txt", "r")
labeldata = labelpath.read()
labellist = labeldata.split("\n")

for i in range(batch_len):
    img_i = data[batch_size * b + i]['img']
    frame_batch[i] = img_i
    frame_batch_ori[i] = data[batch_size * b + i]['orig_img']
    labels.append(labellist[batch_size * b + i])

print(labels)
print("Encoding...")
cond = model.encode(frame_batch.to(device))

#Use random semantic subcode
#seed = np.random.randint(0, 1000000)
#torch.manual_seed(seed)
#cond = torch.randn(1, 512, device=device)

xT = model.encode_stochastic(x=frame_batch.to(device), cond=cond, T=2)

#Use random stochastic encoding
#xT = torch.randn(1, 3, 9, conf.img_size, conf.img_size, device=device)

print("Decoding...")
pred = model.render(noise=xT, cond=cond, T=None)

print("Plotting...")
F = 9
fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
#frame_batch_ori = frame_batch_ori.permute(0, 2, 1, 3, 4)
for i in range(F):
    img = frame_batch_ori[i]
    ax[i].imshow(img.permute(1, 2, 0).cpu())
plt.savefig("viz/ped2_ori_ex_nonema_0.png")

fig, ax = plt.subplots(1, F, figsize=(F * 5, 5))
for i in range(F):
    img = pred[i]
    ax[i].imshow(img.permute(1, 2, 0).cpu())
plt.savefig("viz/ped2_gen_ex_nonema_0.png")

print("DONE!")