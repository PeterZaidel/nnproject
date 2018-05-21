import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import random
from PIL import Image
import copy
import pickle
import sys

from pycocotools.coco import COCO
import skimage.io
import io
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from torch.optim import lr_scheduler

from autocorrect import spell
import nltk
from IPython.display import display
import os
import string

from tqdm import tqdm


DEF_END = '<END>'
DEF_START = '<START>'

dataDir='../../coco_dataset/'
imagesDirTrain = '{}train2017'.format(dataDir)
imagesDirVal = '{}val2017'.format(dataDir)

annTrainFile = '{}annotations/captions_train2017.json'.format(dataDir)
annValFile = '{}annotations/captions_val2017.json'.format(dataDir)

transform = transforms.Compose([transforms.Resize((224, 224)), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

random.seed(1234)



def numpy2image(img_numpy):
    if img_numpy.dtype == np.dtype('float64'):
        img_numpy = (img_numpy*255).astype('uint8')
    return Image.fromarray(img_numpy)


class MSCOCODataset(Dataset):
    """MSCOCO Dataset"""

    def __init__(self, annFile, imagesDir, transform=None, pretrained=False):
        self.coco = COCO(annFile)
        self.imagesDir = imagesDir
        self.imageids = self.coco.getImgIds()
        self.transform = transform
        self.pretrained = pretrained
        self.images = None
        self.imid = None

    def __len__(self):
        return len(self.coco.dataset['images'])

    def __getitem__(self, idx):
        
        if self.pretrained:
            return {'image': self.images[idx], 'id': self.imid[idx]}
        
        imid = self.imageids[idx]
        img_data = self.coco.loadImgs([imid])[0]
        
        img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
        image = skimage.io.imread(img_file_name)
        
        if len(image.shape) != 3:
            return self.__getitem__(0)
        
        image = numpy2image(image)
        if self.transform is not None:
            image = self.transform(image)
            
        sample = {'image': image, 'id': imid}
            
        return sample
    
    def load(self, f_image, f_image_ids):
        self.pretrained = True
        self.images = torch.load(f_image)
        self.imid = torch.load(f_image_ids)
        
        
dtype = torch.cuda.FloatTensor
dtype_2 = torch.cuda.LongTensor

class Caption(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, word_emb):
        super(Caption, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        #self.dec = nn.Linear(self.input_size, self.emb_size)
        #self.lstm = nn.LSTMCell(self.emb_size + 300, self.hidden_size)
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.fc = nn.Sequential(nn.Linear(self.hidden_size, self.output_size), nn.LogSoftmax(1))
        
        self.word_emb = word_emb
        
    def forward(self, X, y, force=True, max_iter=20):
        
        #X = self.dec(X)
        
        C = Variable(torch.zeros(X.size(0), self.hidden_size).type(dtype), requires_grad=False)
        hidden = Variable(torch.zeros(X.size(0), self.hidden_size).type(dtype), requires_grad=False)
        output = []
        
        prev_words = [torch.Tensor(self.word_emb[DEF_START]).view(1, -1) for _ in range(X.size(0))]
        prev_words = Variable(torch.cat(prev_words, 0).type(dtype))
        
        for i in range(max_iter):
            input = torch.cat((X, prev_words), 1)
                
            hidden, C = self.lstm.forward(input, (hidden, C))
            out = self.fc.forward(hidden)
                    
            output.append(out)
            
            if force:
                ids = np.argmax(out.data, 1).numpy()
                prev_words = [torch.Tensor(self.word_emb[ids2words[t]]).view(1, -1) for t in ids]
                prev_words = Variable(torch.cat(prev_words, 0).type(dtype), requires_grad=False)
            else:
                prev_words = [torch.Tensor(self.word_emb[ids2words[int(t)]]).view(1, -1) for t in y[:, i]]
                prev_words = Variable(torch.cat(prev_words, 0).type(dtype), requires_grad=False)
                
        return output
    
    def predict(self, X, max_iter):
        
        #X = self.dec(X)
        
        C = Variable(torch.zeros(X.size(0), self.hidden_size).type(dtype), requires_grad=False)
        hidden = Variable(torch.zeros(X.size(0), self.hidden_size).type(dtype), requires_grad=False)
        output = [[] for _ in range(X.size(0))]
        
        prev_words = [torch.Tensor(self.word_emb[DEF_START]).view(1, -1) for _ in range(X.size(0))]
        prev_words = Variable(torch.cat(prev_words, 0).type(dtype))
        
        for i in range(max_iter):
            input = torch.cat((X, prev_words), 1)
            
            hidden, C = self.lstm.forward(input, (hidden, C))
            out = self.fc.forward(hidden)
            
            ids = np.argmax(out.data, 1).numpy()
            for t in range(X.size(0)):
                output[t].append(ids2words[ids[t]])
            
            prev_words = [torch.Tensor(self.word_emb[ids2words[t]]).view(1, -1) for t in ids]
            prev_words = Variable(torch.cat(prev_words, 0).type(dtype))
            
        return output
    
def ids2sen(ids, train=True):
    ids_dict = None
    if train:
        ids_dict = ids2sen_train
    else:
        ids_dict = ids2sen_test
    
    res = []
    for idx in ids:
        res.append(ids_dict[int(idx)])
    res = torch.cat(res, 0)
    return res


def train(net, loss, optim, learning_rate, epochs, train_loader, test_loader):
    opt = optim(net.parameters())
    train_loss = []
    test_loss = []
    try:
        for it in range(epochs):
            losses = []
            for sample in tqdm(train_loader):
                X = sample['image']
                X = Variable(X.type(dtype))
                ids = sample['id']
                y = Variable(ids2sen(ids, True).type(dtype_2), requires_grad=False) 

                opt.zero_grad()
                state = bool(random.randint(0, 1))
                #state = False
                prediction = net.forward(X, y, state)

                loss_batch = Variable(torch.Tensor([0]).type(dtype))
                for i, out in enumerate(prediction):
                    loss_batch += loss(out, y[:, i])

                losses.append(int(loss_batch.data))
                loss_batch.backward()

                opt.step()
            train_loss.append(np.mean(losses))
            torch.save(net.state_dict(), 'model_resnet_gen.pth')
            
            losses = []
            for sample in tqdm(test_loader):
            
                X = sample['image']
                X = Variable(X.type(dtype))
                ids = sample['id']
                y = Variable(ids2sen(ids, False).type(dtype_2), requires_grad=False) 
                
                state = True
                prediction = net.forward(X, y, state)

                loss_batch = Variable(torch.Tensor([0]).type(dtype))
                for i, out in enumerate(prediction):
                    loss_batch += loss(out, y[:, i])

                losses.append(int(loss_batch.data))
            test_loss.append(np.mean(losses))
            print '\rEposh {}... NLL (Train/Test)   {}/{}'.format(it, train_loss[-1], test_loss[-1])
    except KeyboardInterrupt:
        return

def tokenize(s):
    s = s.lower()
    for t in string.punctuation: 
        s = s.replace(t, ' ')
    return s
    
if __name__ == '__main__':
    
    word_emb = pickle.load(open('word_emb.p', 'r'))
    wrong_ann_ids_train = pickle.load(open('wrong_ann_ids_train.p', 'r'))
    wrong_ann_ids_test = pickle.load(open('wrong_ann_ids_test.p', 'r'))
    
    coco_train = COCO(annTrainFile)
    coco_test = COCO(annValFile)

    words2ids = {}
    ids2words = {}
    for i, key in enumerate(word_emb.keys()):
        words2ids[key] = i
        ids2words[i] = key

    ids2sen_train = {}
    ids = coco_train.getImgIds()
    for idx in ids:
        anns_ids = coco_train.getAnnIds(int(idx))
        ann = None
        for ann_id in anns_ids:
            if ann_id in wrong_ann_ids_train:
                continue
            ann = coco_train.loadAnns(ann_id)[0]['caption']
            ann = tokenize(ann).split()
            break
        p = []
        for word in ann:
            p.append(words2ids[word])
        p += [words2ids[DEF_END] for _ in range(20 - len(p))]
        p = torch.Tensor(p).view(1, -1)
        ids2sen_train[idx] = p

    ids2sen_test = {}
    ids = coco_test.getImgIds()
    for idx in ids:
        anns_ids = coco_test.getAnnIds(int(idx))
        ann = None
        for ann_id in anns_ids:
            if ann_id in wrong_ann_ids_test:
                continue
            ann = coco_test.loadAnns(ann_id)[0]['caption']
            ann = tokenize(ann).split()
            break
        p = []
        for word in ann:
            p.append(words2ids[word])
        p += [words2ids[DEF_END] for _ in range(20 - len(p))]
        p = torch.Tensor(p).view(1, -1)
        ids2sen_test[idx] = p


    train_dataset = MSCOCODataset(annTrainFile, imagesDirTrain, transform)
    test_dataset = MSCOCODataset(annValFile, imagesDirVal, transform)

    train_dataset.load('./train_image_resnet.pth', './train_image_ids_resnet.pth')
    test_dataset.load('./test_image_resnet.pth', './test_image_ids_resnet.pth')

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    loss = nn.NLLLoss()
    optim = torch.optim.Adam

    torch.cuda.set_device(0)
    net = Caption(2048 + 300,  512, len(word_emb), word_emb).cuda()
    net.load_state_dict(torch.load('./model_resnet.pth'))

    train(net, loss, optim, 0.001, 10, train_dataloader, test_dataloader)