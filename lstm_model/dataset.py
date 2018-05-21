#!/usr/bin/env python

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import skimage.io as io
from torch.utils.data import Dataset, DataLoader
import random
from autocorrect import spell
import os

import tqdm

def numpy2image(img_numpy):
    if img_numpy.dtype == np.dtype('float64'):
        img_numpy = (img_numpy*255).astype('uint8')
    return Image.fromarray(img_numpy)



PIC2MANY = 'pic2many'
PIC2RAND = 'pic2rand'
ANN2PIC = 'ann2pic'

class MSCOCODataset(Dataset):
    """MSCOCO Dataset"""
    
    def get_image2anns(self):
        result = []
        for imid in self.imageids:
            annIds = self.coco.getAnnIds(imgIds=imid)
        
            anns_data = self.coco.loadAnns(annIds)
            anns = [ann['caption'] for ann in anns_data]
            result.append({'id': imid, 'anns': anns})
            
        return result
    
    def get_anns(self):
        result = {}
        for annid in tqdm.tqdm(self.annids):
            result[annid] = self.get_ann(annid)
            
        return result
        
    def __init__(self, annFile, imagesDir, transform = None, 
                 mode = 'pic2many', text_transform = None):
        
        self.transform = transform
        self.text_transform = text_transform
        
        self.coco = COCO(annFile)
        
        self.imagesDir = imagesDir
        self.imageids = self.coco.getImgIds()
        self.annids = self.coco.getAnnIds()
        
        self.preload = False
        self.preload_anns = False
        
        
        
        self.mode = mode
        
        self.anns = {}
        self.images_cnn = {}
        
        self.saved_item = self.__getitem__(0)
        
    def __kostyl_create_image_cnn(self):
        self.images_cnn = {}
        self.saved_item = self.__getitem__(0)
        
        
    def preload_anotations(self):
        self.anns = self.get_anns()
        self.preload_anns = True
        
    
    def preload_data(self):
        self.preload = True
        for sample in tqdm.tqdm_notebook(self):        
            self._data_preload.append(sample)
        

    def __len__(self):
        if self.mode == PIC2MANY:
            return len(self.coco.dataset['images'])
        elif self.mode == PIC2RAND:
            return len(self.coco.dataset['images'])
        elif self.mode == ANN2PIC:
            return len(self.coco.dataset['annotations'])

    def get_image(self, imid):
        img_data = self.coco.loadImgs([imid])[0]
        img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
        image = io.imread(img_file_name)
        
        if len(image.shape) != 3:
           return self.get_image(self.imageids[0])
        
        image = numpy2image(image)
        
        if self.transform:
           image = self.transform(image)
        return image
    
    def get_item_data(self, idx):
        
        if self.mode == PIC2MANY or self.mode == PIC2RAND:        
            imid = self.imageids[idx]
            img_data = self.coco.loadImgs([imid])[0]
        
            annIds = self.coco.getAnnIds(imgIds=imid)
        
            anns_data = self.coco.loadAnns(annIds)
            anns = [ann['caption'] for ann in anns_data]
            if self.mode == PIC2RAND:
                anns = anns[random.randint(0,len(anns) - 1 )]
        
            img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
      
            sample = {'id': imid,'anns_ids': annIds, 'img_data': img_data, 
                  'image_file': img_file_name, 'anns': anns}

            return sample
            
        elif self.mode == ANN2PIC:
            annid = self.annids[idx]
            ann_data = self.coco.loadAnns([annid])[0]
            anns = ann_data['caption']
            imid = ann_data['image_id']
            img_data = self.coco.loadImgs([imid])[0]
            
            img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])

      
            sample = {'id': imid, 'anns_ids': [annid],'img_data': img_data, 
                  'image_file': img_file_name, 'anns': anns}

            return sample
        
        
    
    def get_ann(self, annid):
        
        if self.preload_anns == True:
            return self.anns[annid]
        
        ann = self.coco.loadAnns([annid])[0]['caption']
        if self.text_transform:
            ann = self.text_transform(ann)
        return ann
    

    def __getitem__(self, idx):
        
        if self.preload == True:
            return self._data_preload[idx]
        
        item_data = self.get_item_data(idx)
        
        img_file_name = item_data['image_file']
        imid = item_data['id']
        annids = item_data['anns_ids']
        
        image = self.images_cnn.get(imid, None)
        if image is None:
            image = io.imread(img_file_name)
            if len(image.shape) != 3:
                return self.saved_item

            else:
                image = numpy2image(image)
                            
                if self.transform:
                    image = self.transform(image)
                
        
        if self.mode == ANN2PIC or self.mode == PIC2RAND:           
            ann_len = len(self.get_ann(annids[0]))
            sample = {'imid': imid, 'image': image, 'anns': annids[0], 'ann_len': ann_len}
        else:
            ann_len = int(np.array([len(self.get_ann(idx)) for idx in  annids]).max())
            sample = {'imid': imid, 'image': image, 'anns': annids, 'ann_len': ann_len }
        
        return sample
            

            
                
        

