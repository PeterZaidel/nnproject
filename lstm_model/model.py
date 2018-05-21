import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch
from torch import nn
from torch.autograd import Variable
import pandas
from sklearn.preprocessing import MinMaxScaler
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import copy
import shutil

import sys
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from torch.nn.utils.rnn import pack_padded_sequence

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from torch.utils.data import Dataset, DataLoader

from dataset import MSCOCODataset
from torch.optim import lr_scheduler
from autocorrect import spell
import nltk
from IPython.display import display
import os
import gc
from utils import *
from dataset import *

class DecoderRNN(nn.Module):
    def __init__(self, cnn_features_size,  embed_size, hidden_size,
                 vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(cnn_features_size, embed_size),
                                  nn.BatchNorm1d(embed_size, momentum=0.01)).cuda()

        self.embed = nn.Embedding(vocab_size, embed_size).cuda()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True).cuda()
        self.linear = nn.Linear(hidden_size, vocab_size).cuda()
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        self.train_mode()
        features = features.cuda()
        features = self.fc1(features)
        captions = captions.cuda()
        embeddings = self.embed(captions)
        embeddings = torch.cat([features.unsqueeze(1), embeddings], 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        del embeddings, packed, hiddens
        return outputs
    
    def train_mode(self):
        for layer in self.fc1:
            layer.training = True
        self.linear.training = True
        self.lstm.training = True
        self.embed.training = True

    def test_mode(self):
        for layer in self.fc1:
            layer.training = False
        self.linear.training = False
        self.lstm.training = False
        self.embed.training = False

    def sample(self, features, states=None):
        self.test_mode()
        sampled_ids = []
        #inputs = Variable(features.unsqueeze(0))
        features = Variable(features).cuda()
        inputs = self.fc1(features)
        inputs = inputs.unsqueeze(0)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  
            outputs = nn.LogSoftmax()(self.linear(hiddens.squeeze(1)))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)

        return sampled_ids
