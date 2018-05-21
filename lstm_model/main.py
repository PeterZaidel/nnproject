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
from torch.optim import lr_scheduler

from utils import *
import argparse

from model import *

from train_util import *

class Configuration:
    def __init__(self, args):
        self.gpu_device = int(args.gpu)
        self.epochs = int(args.epochs)
        self.checkpoint_start = args.checkpoint_start
        self.checkpoint_file = args.checkpoint
        self.coco_dir = args.coco_dataset_dir
        self.mode = args.mode
        self.batch_size = args.batch

print("HELLO")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default='train',
    help="train or test mode")

parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU number for training")

parser.add_argument(
    "--batch",
    type=int,
    default=128,
    help="Batch size")

parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="Epochs number")

parser.add_argument(
    "--checkpoint_start",
    type=bool,
    default=False,
    help="Start training from checkpoint")

parser.add_argument(
    "--checkpoint",
    type=str,
    default='checkpoints/checkpoint.pth.tar',
    help="path to checkpoint file")

parser.add_argument(
    "--coco_dataset_dir",
    type=str,
    default='/mnt/disk/p.zaydel_OLOLO/ProjectNeuralNets/coco_dataset/',
    help="directory with coco dataset")



args = parser.parse_args()
print(args)
conf = Configuration(args)


gpu_device = conf.gpu_device

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(conf.gpu_device)

trainDataset, testDataset = load_or_create_datasets()
trainDataset.mode = 'ann2pic'
testDataset.mode = 'ann2pic'

print(len(trainDataset))
print(len(testDataset))

#words2ids, ids2words = load_or_create_dictionaries(trainDataset, testDataset)

vocab = load_or_create_vocab(trainDataset, testDataset)

cnn_features_size = trainDataset[0]['image'].shape[0]
# cnn_features_size = models.resnet152().fc.in_features
cap_net = DecoderRNN(cnn_features_size=cnn_features_size, embed_size=512, hidden_size=1024, vocab_size=len(vocab), num_layers=1, max_seq_length=20)

trainDataLoader = DataLoader(trainDataset, batch_size = conf.batch_size, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size = conf.batch_size, shuffle=True)

if conf.mode == 'train':
    train(cap_net, trainDataLoader, testDataLoader, trainDataset, testDataset, vocab, epochs=conf.epochs,
          load_checkpoint=conf.checkpoint_start, checkpoint_file=conf.checkpoint_file, criterion=nn.CrossEntropyLoss(), optim=torch.optim.Adam)
elif conf.mode == 'test':
    if conf.checkpoint_file is None or conf.checkpoint_file == '':
        test_network(cap_net, testDataset, vocab, load_checkpoint=True, count = -1)
    else:
        test_network(cap_net, testDataset, vocab, load_checkpoint=True, checkpoint_file=conf.checkpoint_file, count = -1)
