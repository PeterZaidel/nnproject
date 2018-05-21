
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


class LSTM_W2V_Net_Cnn_Preload(nn.Module):

    def __init__(self,  image_size, image_features_size, word_embedding, words2ids, ids2words,
                 lstm_hidden_size = 256,
                 word_embedding_size =  300,
                 cnn = models.alexnet(pretrained=True).features,
                 start_symbol = DEF_START,
                 end_symbol = DEF_SEND
                  ):
        """Init NN
            image_size - size of input image.
            lstm_hidden_size - size of cnn features output
            image_features_size - size of image features vector
            word_embedding - pretrained word embedding model
            words2ids - dictionary word -> id
            ids2words - dictionary id -> word
            cnn - pretrained cnn net (alexnet, vgg and other)
            start_symbol - symbol starting sequence
            end_symbol - symbol ending sequence
        """

        super(LSTM_W2V_Net_Cnn_Preload, self).__init__()
        self.image_size = image_size
        self.image_features_size = image_features_size
        #self.cnn = cnn
     #   self.cnn_comp_features = cnn_comp_features

        self.vocab_size = len(words2ids)
        print(self.vocab_size)

        self.word_embedding_size = word_embedding_size
        self.word_embedding = word_embedding

        self.words2ids = words2ids
        self.ids2words = ids2words

        self.start_symbol = start_symbol
        self.start_symbol_embed = torch.from_numpy(self.word_embedding[self.start_symbol])

        self.end_symbol = end_symbol
        self.end_symbol_embed = torch.from_numpy(self.word_embedding[self.end_symbol])

#         self.sentence_end_symbol = sentence_end_symbol
#         self.sentence_end_symbol_id = self.words2ids[self.sentence_end_symbol]

#         if sentence_end_embed is not None:
#             self.sentence_end_embed = sentence_end_embed
#         else:
#             self.sentence_end_embed = word_embeding['.']

        #self.max_sentence_len = max_sentence_len

        self.lstm_hidden_size = lstm_hidden_size

        # self.fc1 = nn.Sequential( nn.BatchNorm1d(self.image_features_size),
        #                           nn.Linear(self.image_features_size, int(self.image_features_size/2)),
        #                           nn.Dropout(0.001),
        #                           nn.ReLU(),
        #                           nn.Linear(int(self.image_features_size/2), int(self.image_features_size/4) ),
        #                           nn.Dropout(0.001),
        #                           nn.ReLU(),
        #                           nn.Linear(int(self.image_features_size/4), self.lstm_hidden_size),
        #                           nn.BatchNorm1d(self.lstm_hidden_size)
        #                         ).cuda()

        self.fc1 = nn.Sequential( nn.Linear(self.image_features_size, self.lstm_hidden_size)).cuda()

        self.lstm_cell = nn.LSTMCell(self.lstm_hidden_size + self.word_embedding_size,
                                     self.lstm_hidden_size).cuda()

        self.fc2 = nn.Sequential(nn.Linear(self.lstm_hidden_size, self.vocab_size),
                                  nn.LogSoftmax() ).cuda()


#         self.lstm = nn.LSTM(self.lstm_hidden_size , word_embedding_size)



#     def freeze_cnn(self):
#         for param in self.cnn.parameters():
#             param.requires_grad = False

#     def unfreeze_cnn(self):
#         for param in self.cnn.parameters():
#             param.requires_grad = True

    def set_mode(self, mode):
        if mode == 'train':
            for layer in self.fc1:
                layer.training = True

            for layer in self.fc2:
                layer.training = True
        elif mode == 'test':
            for layer in self.fc1:
                layer.training = False

            for layer in self.fc2:
                layer.training = False


    # word ids -> words embeddings
    def ids_to_embed(self, word_ids):
        result = []

        for i in range(word_ids.shape[0]):
            w = self.ids2words[word_ids[i].data[0]]

            emb = torch.from_numpy(self.word_embedding[w]).float()
            result.append(emb)

        return torch.stack(result)

    def forward(self, X, max_sentence_len, label=None, mode = 'train'):
        if mode == 'train':
            return self.forward_train(X, max_sentence_len, label)
        else:
            return self.forward_test(X, max_sentence_len)


    def forward_train(self, X, max_sentence_len, label):
        batch_size = X.shape[0]
        X = X.cuda()
        X = self.fc1(X)

        prevWord = Variable(self.start_symbol_embed.repeat(batch_size, 1), requires_grad=True).cuda()
        lstm_input = torch.cat([X, prevWord], dim = 1)

        result = []

        h_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()

        start_word_id = self.words2ids[DEF_START]
        start_word_one_hot = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            start_word_one_hot[i, start_word_id] = 1
        start_word_one_hot = Variable(start_word_one_hot)

        result.append(start_word_one_hot)

        for idx in range(1, max_sentence_len):
            torch.cuda.empty_cache()

            h_t, c_t = self.lstm_cell.forward(lstm_input, (h_t, c_t))

            probs = self.fc2.forward(h_t)
            probs = probs.cpu()

            # top_word_ids = probs.max(1)[1].cpu()
            # embeds = self.ids_to_embed(top_word_ids)
            # embeds = Variable(embeds)
            # del lstm_input, top_word_ids

            embeds = self.ids_to_embed(label[idx])
            embeds = Variable(embeds)

            lstm_input = torch.cat([X, embeds.cuda()], dim = 1).cuda()
            del embeds
            result.append(probs.cpu())
            del probs


        result = torch.stack(result, dim = 1).cpu()

        del X, h_t, c_t, lstm_input, prevWord
        torch.cuda.empty_cache()

        return result


    def forward_test(self, X, max_sentence_len):
        batch_size = X.shape[0]
        X = X.cuda()
        X = self.fc1(X)

        prevWord = Variable(self.start_symbol_embed.repeat(batch_size, 1), requires_grad=True).cuda()
        lstm_input = torch.cat([X, prevWord], dim = 1)

        result = []

        h_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()

        start_word_id = self.words2ids[DEF_START]
        start_word_one_hot = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            start_word_one_hot[i, start_word_id] = 1
        start_word_one_hot = Variable(start_word_one_hot)

        result.append(start_word_one_hot)

        for idx in range(max_sentence_len-1):
            torch.cuda.empty_cache()

            h_t, c_t = self.lstm_cell.forward(lstm_input, (h_t, c_t))

            probs = self.fc2.forward(h_t)
            probs = probs.cpu()

            top_word_ids = probs.max(1)[1].cpu()
            embeds = self.ids_to_embed(top_word_ids)
            embeds = Variable(embeds)
            del lstm_input, top_word_ids

            lstm_input = torch.cat([X, embeds.cuda()], dim = 1).cuda()
            del embeds
            result.append(probs.cpu())
            del probs

        result = torch.stack(result, dim = 1).cpu()

        del X, h_t, c_t, lstm_input, prevWord
        torch.cuda.empty_cache()

        return result


class LSTM_2_Layer(nn.Module):
    def __init__(self, image_size, image_features_size, word_embedding, words2ids, ids2words,
                 lstm_hidden_size=256,
                 word_embedding_size=300,
                 cnn=models.alexnet(pretrained=True).features,
                 start_symbol=DEF_START,
                 end_symbol=DEF_SEND
                 ):
        """Init NN
            image_size - size of input image.
            lstm_hidden_size - size of cnn features output
            image_features_size - size of image features vector
            word_embedding - pretrained word embedding model
            words2ids - dictionary word -> id
            ids2words - dictionary id -> word
            cnn - pretrained cnn net (alexnet, vgg and other)
            start_symbol - symbol starting sequence
            end_symbol - symbol ending sequence
        """

        super(LSTM_W2V_Net_Cnn_Preload, self).__init__()
        self.image_size = image_size
        self.image_features_size = image_features_size
        # self.cnn = cnn
        #   self.cnn_comp_features = cnn_comp_features

        self.vocab_size = len(words2ids)
        print(self.vocab_size)

        self.word_embedding_size = word_embedding_size
        self.word_embedding = word_embedding

        self.words2ids = words2ids
        self.ids2words = ids2words

        self.start_symbol = start_symbol
        self.start_symbol_embed = torch.from_numpy(self.word_embedding[self.start_symbol])

        self.end_symbol = end_symbol
        self.end_symbol_embed = torch.from_numpy(self.word_embedding[self.end_symbol])

        #         self.sentence_end_symbol = sentence_end_symbol
        #         self.sentence_end_symbol_id = self.words2ids[self.sentence_end_symbol]

        #         if sentence_end_embed is not None:
        #             self.sentence_end_embed = sentence_end_embed
        #         else:
        #             self.sentence_end_embed = word_embeding['.']

        # self.max_sentence_len = max_sentence_len

        self.lstm_hidden_size = lstm_hidden_size

        # self.fc1 = nn.Sequential( nn.BatchNorm1d(self.image_features_size),
        #                           nn.Linear(self.image_features_size, int(self.image_features_size/2)),
        #                           nn.Dropout(0.001),
        #                           nn.ReLU(),
        #                           nn.Linear(int(self.image_features_size/2), int(self.image_features_size/4) ),
        #                           nn.Dropout(0.001),
        #                           nn.ReLU(),
        #                           nn.Linear(int(self.image_features_size/4), self.lstm_hidden_size),
        #                           nn.BatchNorm1d(self.lstm_hidden_size)
        #                         ).cuda()

        self.fc1 = nn.Sequential(nn.Linear(self.image_features_size, self.lstm_hidden_size)).cuda()

        self.lstm_cell = nn.LSTMCell(self.lstm_hidden_size + self.word_embedding_size,
                                     self.lstm_hidden_size).cuda()

        self.fc2 = nn.Sequential(nn.Linear(self.lstm_hidden_size, self.vocab_size),
                                 nn.LogSoftmax()).cuda()

    #         self.lstm = nn.LSTM(self.lstm_hidden_size , word_embedding_size)



    #     def freeze_cnn(self):
    #         for param in self.cnn.parameters():
    #             param.requires_grad = False

    #     def unfreeze_cnn(self):
    #         for param in self.cnn.parameters():
    #             param.requires_grad = True

    def set_mode(self, mode):
        if mode == 'train':
            for layer in self.fc1:
                layer.training = True

            for layer in self.fc2:
                layer.training = True
        elif mode == 'test':
            for layer in self.fc1:
                layer.training = False

            for layer in self.fc2:
                layer.training = False

    # word ids -> words embeddings
    def ids_to_embed(self, word_ids):
        result = []

        for i in range(word_ids.shape[0]):
            w = self.ids2words[word_ids[i].data[0]]

            emb = torch.from_numpy(self.word_embedding[w]).float()
            result.append(emb)

        return torch.stack(result)

    def forward(self, X, max_sentence_len, label=None, mode='train'):
        if mode == 'train':
            return self.forward_train(X, max_sentence_len, label)
        else:
            return self.forward_test(X, max_sentence_len)

    def forward_train(self, X, max_sentence_len, label):
        batch_size = X.shape[0]
        X = X.cuda()
        X = self.fc1(X)

        prevWord = Variable(self.start_symbol_embed.repeat(batch_size, 1), requires_grad=True).cuda()
        lstm_input = torch.cat([X, prevWord], dim=1)

        result = []

        h_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()

        start_word_id = self.words2ids[DEF_START]
        start_word_one_hot = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            start_word_one_hot[i, start_word_id] = 1
        start_word_one_hot = Variable(start_word_one_hot)

        result.append(start_word_one_hot)

        for idx in range(1, max_sentence_len):
            torch.cuda.empty_cache()

            h_t, c_t = self.lstm_cell.forward(lstm_input, (h_t, c_t))

            probs = self.fc2.forward(h_t)
            probs = probs.cpu()

            # top_word_ids = probs.max(1)[1].cpu()
            # embeds = self.ids_to_embed(top_word_ids)
            # embeds = Variable(embeds)
            # del lstm_input, top_word_ids

            embeds = self.ids_to_embed(label[idx])
            embeds = Variable(embeds)

            lstm_input = torch.cat([X, embeds.cuda()], dim=1).cuda()
            del embeds
            result.append(probs.cpu())
            del probs

        result = torch.stack(result, dim=1).cpu()

        del X, h_t, c_t, lstm_input, prevWord
        torch.cuda.empty_cache()

        return result

    def forward_test(self, X, max_sentence_len):
        batch_size = X.shape[0]
        X = X.cuda()
        X = self.fc1(X)

        prevWord = Variable(self.start_symbol_embed.repeat(batch_size, 1), requires_grad=True).cuda()
        lstm_input = torch.cat([X, prevWord], dim=1)

        result = []

        h_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()
        c_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False).cuda()

        start_word_id = self.words2ids[DEF_START]
        start_word_one_hot = torch.zeros(batch_size, self.vocab_size)
        for i in range(batch_size):
            start_word_one_hot[i, start_word_id] = 1
        start_word_one_hot = Variable(start_word_one_hot)

        result.append(start_word_one_hot)

        for idx in range(max_sentence_len - 1):
            torch.cuda.empty_cache()

            h_t, c_t = self.lstm_cell.forward(lstm_input, (h_t, c_t))

            probs = self.fc2.forward(h_t)
            probs = probs.cpu()

            top_word_ids = probs.max(1)[1].cpu()
            embeds = self.ids_to_embed(top_word_ids)
            embeds = Variable(embeds)
            del lstm_input, top_word_ids

            lstm_input = torch.cat([X, embeds.cuda()], dim=1).cuda()
            del embeds
            result.append(probs.cpu())
            del probs

        result = torch.stack(result, dim=1).cpu()

        del X, h_t, c_t, lstm_input, prevWord
        torch.cuda.empty_cache()

        return result





