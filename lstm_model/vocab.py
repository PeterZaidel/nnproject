
import numpy as np
from torch import nn

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
from collections import Counter
from pycocotools.coco import COCO
import argparse


DEF_SEND = '<SEND>'
DEF_START = '<START>'
DEF_UNK = '<UNK>'


class Vocabulary(object):
    def __init__(self):
        self.words2ids = {}
        self.ids2words = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.words2ids:
            self.words2ids[word] = self.idx
            self.ids2words[self.idx] = word
            self.idx += 1

    def create_from_texts(self, Texts, threshold = 4):
        counter = Counter()
        for tokens in Texts:
            counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= threshold]
        for w in words:
            self.add_word(w)

        # uniqwords = list(set([w for ann in Texts for w in ann]))
        # self.words2ids = dict(zip(uniqwords, range(len(uniqwords))))
        # self.ids2words = dict(zip(range(len(uniqwords)), uniqwords))
        self.add_word(DEF_SEND)
        self.add_word(DEF_START)
        self.add_word(DEF_UNK)

        print("Vocab Len: ", len(self.words2ids))

    def __call__(self, word):
        if not word in self.words2ids:
            return self.words2ids[DEF_UNK]
        return self.words2ids[word]

    def get_word(self, id):
        if not id in self.ids2words:
            return self.get_word(self(DEF_UNK))
        return self.ids2words[id]

    def __len__(self):
        return len(self.words2ids)


def build_vocab(annFile, threshold, vocab= None):
    coco = COCO(annFile)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in tqdm(enumerate(ids)):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    words = [word for word, cnt in counter.items() if cnt >= threshold]
    

    # Create a vocab wrapper and add some special tokens.
    if vocab == None:
        vocab = Vocabulary()

    vocab.add_word(DEF_START)
    vocab.add_word(DEF_SEND)
    vocab.add_word(DEF_UNK)

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(args.caption_path_train, threshold=args.threshold)
    vocab = build_vocab(args.caption_path_test, threshold=args.threshold, vocab=vocab)

    torch.save(vocab, args.vocab_path)
    vocab_path = args.vocab_path
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    from utils import annTrainFile, annValFile, VOCAB_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path_train', type=str,
                        default= annTrainFile,
                        help='path for train annotation train file')

    parser.add_argument('--caption_path_test', type=str,
                        default= annValFile,
                        help='path for train annotation train file')


    parser.add_argument('--vocab_path', type=str, default=VOCAB_FILE,
                        help='path for saving vocabulary wrapper')

    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')

    args = parser.parse_args()
    main(args)
