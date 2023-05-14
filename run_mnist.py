from templates import *
from templates_latent import *

if __name__ == '__main__':
    # train the autoenc moodel
    # this can be run on 2080Ti's.
   
    gpus = [0,1,2,3]
    conf = mnist_autoenc()
    conf.net_ch = 32
    #conf.sample_size = 8
    train(conf, gpus=gpus)