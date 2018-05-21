
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

from autocorrect import spell
import nltk
from dataset import MSCOCODataset
from tqdm import tqdm

from gensim.models import Word2Vec
import os
import gc
from vocab import Vocabulary
import vocab

DEF_SEND = '<SEND>'
DEF_START = '<START>'


TRAIN_DATSET_FILE = 'datasets/traindataset_resnet.tar.gz'
TEST_DATSET_FILE =  'datasets/testdataset_resnet.tar.gz'

WORD_EMBED_FILE = 'texts/word_embeding.tar.gz'

VOCAB_FILE = 'texts/vocab.tar.gz'


dataDir='/mnt/disk/p.zaydel_OLOLO/ProjectNeuralNets/coco_dataset/'
imagesDirTrain = '{}train2017/train2017'.format(dataDir)
imagesDirVal = '{}val2017/val2017'.format(dataDir)

annTrainFile = '{}/annotations_trainval2017/annotations/captions_train2017.json'.format(dataDir)
annValFile = '{}/annotations_trainval2017/annotations/captions_val2017.json'.format(dataDir)


transform_tensor = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                           ])
transform_to224 = transforms.Compose([transforms.Resize((224, 224)),
                                      transform_tensor
                                     ])
transform_to500 = transforms.Compose([ transforms.Resize((500, 500)),
                                       transform_tensor
                                           ])

transform_to256 = transforms.Compose([ transforms.Resize((256, 256)),
                                      transform_tensor
                                           ])



def load_or_create_datasets():
    if os.path.exists(TRAIN_DATSET_FILE):
        print("loading train dataset...")
        trainDataset = torch.load(TRAIN_DATSET_FILE)
        print('train dataset loaded!')
    else:
        cnn_model = models.alexnet(pretrained=True)
        trainDataset = MSCOCODataset(annTrainFile, imagesDirTrain, transform=transform_to224, mode='ann2pic')
        save_prepared_dataset(trainDataset, TRAIN_DATSET_FILE, cnn_model)

    if os.path.exists(TEST_DATSET_FILE):
        print("loading test dataset...")
        testDataset = torch.load(TEST_DATSET_FILE)
        print('test dataset loaded!')
    else:
        cnn_model = models.alexnet(pretrained=True)
        testDataset = MSCOCODataset(annValFile, imagesDirVal, transform=transform_to224, mode='ann2pic')
        save_prepared_dataset(testDataset, TEST_DATSET_FILE, cnn_model)
    return trainDataset, testDataset


def load_or_create_vocab(trainDataset=None, testDataset=None):
    Texts = list(trainDataset.anns.values()) + list(testDataset.anns.values())
    if os.path.exists(VOCAB_FILE):
        print("loading vocab")
        vocab = torch.load(VOCAB_FILE)
        print("vocab loaded")
        return vocab
    else:
        vocab = Vocabulary()
        vocab.create_from_texts(Texts)
        return vocab

# def load_or_create_embedings(trainDataset, testDataset):
#     Texts = list(trainDataset.anns.values()) + list(testDataset.anns.values())
#     if os.path.exists(WORD_EMBED_FILE):
#         print("loading words embedding")
#         word_embeding = torch.load(WORD_EMBED_FILE)
#         print("words embedding loaded")
#     else:
#         print("creating words embedding......")
#         word_embeding = train_word_to_vec_gensim(Texts, embed_size=300)
#         print("saving words embedding")
#         torch.save(word_embeding, WORD_EMBED_FILE)
#     return word_embeding


def split_text2words(text):
    symbs_to_replace = ['.', ',', '/', '-', ':', '{', '}', '[', ']', ]
    for smb in symbs_to_replace:
        text = text.replace(smb, ' ')


    words = nltk.word_tokenize(text.lower())

    # for idx in range(len(words)):
    #     words[idx] = spell(words[idx])

    words = [DEF_START] + words + [DEF_SEND]

    return words



# def train_word_to_vec_gensim(anns, embed_size = 300):
#     Texts = list(anns)
#     model = Word2Vec(Texts, size = embed_size, workers = 7, min_count = 0)
#     return model



# # calculates dimension of alexnet convolutions layers output
# def get_alexnet_features_dim(imsize):
#     adim = int(np.round( 3*0.01*imsize - 1))
#     return 1*256*adim*adim

class prepareDataset(Dataset):      
       def __init__(self, dataset):
           self.dataset = dataset
           self.imageids = dataset.imageids

       def __len__(self):
           return len(self.imageids)
       def __getitem__(self, idx):
           imid = self.imageids[idx]
           image = self.dataset.get_image(imid)
           return {'imid': imid , 'image': image}

def save_prepared_dataset(dataset, filename, cnn_model = models.alexnet(pretrained=True).features):

    print("preparing images...")
    
    prepareDS = prepareDataset(dataset)
    prepDL = DataLoader(prepareDS, batch_size = 16, shuffle=False)
    
    counter = 1
    print("start")
    for sample in tqdm(prepDL):
        
        counter += 1
        imids = sample['imid']
        images = Variable(sample['image']).cuda()
        batch_size = imids.shape[0]
        cnn_res = cnn_model(images)

        for idx in range(cnn_res.shape[0]):
            im_vec = cnn_res[idx].data.view(-1).cpu()
            imid = imids[idx]
            dataset.images_cnn[imid] = im_vec

        if counter % 100:
           torch.cuda.empty_cache()
           gc.collect()
        
    
   # for idx in tqdm(range(len(dataset.imageids))):
   #     imid = dataset.imageids[idx]
   #     image = dataset.get_image(imid)
   #     var = Variable(image.unsqueeze(0)).cuda()
   #     cnn_res = cnn_model(var)
   #     cnn_res = cnn_res.cpu()
   #     dataset.images_cnn[imid] = cnn_res.data.view(-1) 
    #    if idx % 200:
    #        torch.cuda.empty_cache()

    print("preparing annotations...")
    dataset.text_transform = split_text2words
    dataset.preload_anotations()
    dataset.text_transform = None

    torch.save(dataset, filename)
    print("Dataset saved in {}".format(filename))


def load_anns(dataset, annids, max_len, prepare = None):
    '''
       dataset - MSCOCODataset
       annids -  tensor or numpy array
       max_len - maximum len of sentence. If None computes from dataset
       prepare - None or function to prepare each word, returns 1-dim tensor

       return Pytorch Tensor [len(annids) x max_sentence_len x prepare(word).shape[0] ]
    '''
    result = []

    if prepare is None:
        prepare = lambda w: w
        #prepare = lambda w: word_embeding[w]

    for i in range(annids.shape[0]):
        words = dataset.get_ann(annids[i])
        ann_res = []

        for idx in range(max_len):
            if idx < len(words):
                w = words[idx]
            else:
                w = DEF_SEND

            ann_res.append(prepare(w))
        ann_res = torch.from_numpy(np.array(ann_res)).float()
        result.append(ann_res)

    return torch.stack(result)
