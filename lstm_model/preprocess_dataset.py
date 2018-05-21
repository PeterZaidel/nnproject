from torch import optim
from torch import nn

import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from dataset import MSCOCODataset

import os
from utils import *


import argparse


dataDir='/mnt/disk/p.zaydel_OLOLO/ProjectNeuralNets/coco_dataset/'
imagesDirTrain = '{}train2017/train2017'.format(dataDir)
imagesDirVal = '{}val2017/val2017'.format(dataDir)

annTrainFile = '{}/annotations_trainval2017/annotations/captions_train2017.json'.format(dataDir)
annValFile = '{}/annotations_trainval2017/annotations/captions_val2017.json'.format(dataDir)


TRAIN_DATSET_FILE = 'datasets/traindataset_resnet.tar.gz'
TEST_DATSET_FILE = 'datasets/testdataset_resnet.tar.gz'

parser = argparse.ArgumentParser()

parser.add_argument(
    "--gpu",
    type=int,
    default=1,
    help="GPU number for training")

parser.add_argument(
    "--cnn",
    type=str,
    default='resnet',
    help="CNN for preprocess resnet/alexnet/vgg/inception")

args = parser.parse_args()

#model_cnn = list(models.resnet152(pretrained=True).children())[:-1]

gpu_device = int(args.gpu)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(gpu_device)

if args.cnn == 'alexnet':
    model_cnn = models.alexnet(pretrained=True).features.cuda()
elif args.cnn == 'vgg':
    model_cnn = models.vgg16(pretrained=True).features.cuda()
elif args.cnn == 'inception':
    modules = list(models.inception_v3(pretrained=True).children())[:-1]
    model_cnn=nn.Sequential(*modules).cuda()
elif args.cnn == 'resnet':
    modules = list(models.resnet152(pretrained=True).children())[:-1]
    model_cnn = nn.Sequential(*modules).cuda()





trainDataset = MSCOCODataset(annTrainFile, imagesDirTrain, transform=transform_to256, mode='pic2rand')
save_prepared_dataset(trainDataset, TRAIN_DATSET_FILE, model_cnn)

testDataset = MSCOCODataset(annValFile, imagesDirVal, transform=transform_to256, mode='pic2rand')
save_prepared_dataset(testDataset, TEST_DATSET_FILE, model_cnn)

