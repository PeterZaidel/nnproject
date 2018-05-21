import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import optim
import torch
from torch import nn
from torch.autograd import Variable

import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from dataset import MSCOCODataset
from torch.optim import lr_scheduler
from autocorrect import spell
import nltk
from IPython.display import display
import os
import sys
from tqdm import tqdm
from tqdm import tqdm_notebook

import pickle
import gc
from model import *
from utils import *


TRAIN_LOG_FILE = "train_log_1.txt"
TRAIN_PLT_FILE = 'train_plt.png'
CHECKPOINT_FILE = 'checkpoints/checkpoint.pth.tar'


TEST_LOG_FILE = 'test_log.txt'
RESULT_FILE = 'test_result.tar.gz'

def save_checkpoint(state, is_best, filename=CHECKPOINT_FILE):
    torch.save(state, filename)

def open_checkpoint(network, optimizer, is_best = False, filename = CHECKPOINT_FILE):
    print("LOADING CHECKPOINT...")

    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    network.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("LOADED CHECKPOINT EPOCH: {}".format(epoch))

def test_network_on_sid(network, testDataset, vocab, sample_idx = 3455):
    sample = testDataset[sample_idx]
    ann_id = sample['anns']
    ann = testDataset.get_ann(ann_id)
    im_id = sample['imid']
    features = sample['image']
    max_len = sample['ann_len']

    pred = network.sample(features.unsqueeze(0))
    
    wids = pred[0]

    result = []
    for i in range(wids.shape[0]):
        wid = wids[i].data[0]
        word = vocab.get_word(wid)
        result.append(word)

    return {'res': result, 'ann_id': ann_id, 'imid': im_id, 'ann': ann, 'max_len': max_len}


def test_network(network, testDataset, vocab, epoch, result_file = RESULT_FILE):
    print("\nSTART TESTING\n")
    sample_ids = []

    for i in range(len(testDataset)):
        sample_ids.append(i)

    log_file = open(str(epoch) + "_" + TEST_LOG_FILE, 'w')

    result_dict = {}

    for sampleid in tqdm(sample_ids):
        res = test_network_on_sid(network, testDataset, vocab,  sample_idx=sampleid)
        log_file.write(str(res))
        log_file.write('\n')

        result_dict[res['imid']] = res

    pickle.dump(result_dict, open(str(epoch) + '_' + result_file, 'wb'), protocol=2)

    print("TEST ENDED!!\n")

def train(network, train_dataloader, test_dataloader, trainDataset, testDataset, vocab,
          epochs, learning_range = 0.001,
          load_checkpoint = False, checkpoint_file = CHECKPOINT_FILE,
          criterion = nn.CrossEntropyLoss(), optim=torch.optim.Adam):

    print("TRAIN STARTED!")
    log_file = open(TRAIN_LOG_FILE,'w')

    train_loss_epochs = []
    test_loss_epochs = []
    optimizer = optim(network.parameters(), lr = learning_range)
    best_test_score = 10**6


    if load_checkpoint:
        open_checkpoint(network, optimizer, is_best=False, filename=checkpoint_file)
    try:
        for epoch in range(epochs):

            train_loss_sum = 0.0
            train_loss_count = 0

            sample_id = 0
            print("Epoch: {} Training".format(epoch + 1))
            for sample in tqdm(train_dataloader):

               sample_id += 1
               torch.cuda.empty_cache()

               features = sample['image']
               ann_ids = sample['anns']
               batch_size = features.shape[0]
               lengths = sample['ann_len']
               max_len = lengths.max()
               
               captions = load_anns(trainDataset, ann_ids,  max_len, prepare=lambda w: vocab(w))
               captions = captions.long()


               lengths, perm_index = lengths.sort(0, descending=True)
               lengths = lengths.numpy()
               captions = captions[perm_index]
               features = features[perm_index]

               captions = Variable(captions)
               features = Variable(features)
               targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
               

               outputs = network.forward(features, captions, lengths)
               targets = targets.cuda()
#               outputs = outputs.cpu()
#               targets = targets.cpu()

               loss_batch = criterion(outputs, targets)
               train_loss_sum += loss_batch.data[0]
               train_loss_count += 1.0

               loss_batch.backward()
               optimizer.step()
               optimizer.zero_grad()
               del features, captions, loss_batch, sample, outputs, targets, lengths
               if sample_id % 200 == 0:
                  gc.collect()

            gc.collect()
            train_loss_epochs.append(train_loss_sum/train_loss_count)

            test_loss_sum = 0.0
            test_loss_count = 0
            sample_id = 0
            torch.cuda.empty_cache()

            print("Epoch: {} Testing".format(epoch + 1))
            for sample in tqdm(test_dataloader):
                sample_id += 1
                torch.cuda.empty_cache()

                features = sample['image']
                ann_ids = sample['anns']
                batch_size = features.shape[0]
                lengths = sample['ann_len']
                max_len = lengths.max()

                captions = load_anns(testDataset, ann_ids, max_len, prepare=lambda w: vocab(w))
                captions = captions.long()

                lengths, perm_index = lengths.sort(0, descending=True)
                lengths = lengths.numpy()
                captions = captions[perm_index]
                features = features[perm_index]

                captions = Variable(captions).cuda()
                features = Variable(features).cuda()
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                outputs = network.forward(features, captions, lengths)
                targets = targets.cuda()
 #               outputs = outputs.cpu()
#              targets = targets.cpu()

                loss_batch = criterion(outputs, targets)
                test_loss_sum += loss_batch.data[0]
                test_loss_count += 1.0

                del features, captions, loss_batch, sample, outputs, targets, lengths
                if sample_id % 200 == 0:
                    gc.collect()



            test_loss_epochs.append(test_loss_sum/(test_loss_count))

            test_network(network, testDataset, vocab, epoch)

            is_best = test_loss_epochs[-1] < best_test_score
            best_test_score = min(test_loss_epochs[-1], best_test_score)
            save_checkpoint({
                            'net': network,
                            'epoch': epoch + 1,
                            'state_dict': network.state_dict(),
                            'best_test_score': best_test_score,
                            'optimizer' : optimizer.state_dict(),
                            }, is_best, filename = 'checkpoints/checkpoint_{}.pth.tar'.format(epoch + 1))
            log_file.write('\rEpoch {0}... (Train/Test) Loss: {1:.3f}/{2:.3f}\n'.format(epoch, train_loss_epochs[-1], test_loss_epochs[-1]))

            sys.stdout.write('\rEpoch {0}... (Train/Test) Loss: {1:.3f}/{2:.3f}\n'.format(
                                                epoch, train_loss_epochs[-1], test_loss_epochs[-1]))
            gc.collect()

    except KeyboardInterrupt:
        pass
    # plt.figure(figsize=(12, 5))
    # plt.plot(train_loss_epochs[1:], label='Train')
    # plt.plot(test_loss_epochs[1:], label='Test')
    # plt.xlabel('Epochs', fontsize=16)
    # plt.ylabel('Loss', fontsize=16)
    # plt.legend(loc=0, fontsize=16)
    # plt.grid('on')
    # plt.savefig(TRAIN_PLT_FILE)
    #
    gc.collect()

    print("TRAIN ENDED!")
