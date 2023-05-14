from templates import *
from templates_latent import *
import os

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
    gpus = [0,1,2,3]
    #print(os.listdir('/home/jy2k16/diffae/train_raw_flows')[0])
    #print(len(os.listdir('/home/jy2k16/diffae/train_raw_flows')))
    conf = shanghaitechflows_autoenc()
    train(conf, gpus=gpus)