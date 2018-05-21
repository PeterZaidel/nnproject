import pickle
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import argparse


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')



def score(num, DIR):
    print("Testing results on epoch ", num, " in DIR=",DIR)
    print("Loading coco annotations")
    dataDir='.'
    dataType='val2014'
    algName = 'fakecap'
    annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
    subtypes=['results', 'evalImgs', 'eval']
    [resFile, evalImgsFile, evalFile]= \
    ['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName,subtype) for subtype in subtypes]
    coco_anns = COCO(annFile)
    print("COCO anns imported")
    
    
    
    path = DIR+ str(num)+'_test_result.tar.gz'
    save = pickle.load(open(path))
    cocoRes = {}
    coco = {}
    for key, val in save.items():
        reslst = val[u'res']
        res = []
        for data in reslst:
            if data!=u'<SEND>':
                res.append(data)
            else:
                break
        res = res[1:]
        #print "RES: ",reslst
        #print "ANN: ", val[u'ann']
        #res = [word for word in res if word!=u'<SEND>'][1:]
        #print "RES FIXED: ", res
        
         
        if len(res) == 0:
            res = [u'a'] #just not to be empty, and it has low low idf
        cocoRes[key] = [{u'caption':' '.join(res)}]
        
        #coco[key] = [{u'caption':' '.join(val[u'ann'][1:-1])}]
        coco[key] = coco_anns.imgToAnns[key]
    print 'examples'
    for key in coco.keys()[:5]:
        print "IMG_NUM=",key
        print "Annotation: ", '\n'.join([coco[key][i][u'caption'] for i in range(len(coco[key]))])
        print "Generated data: ", ' '.join(save[key][u'res'])
        print "Cleared generation: ", cocoRes[key][0][u'caption']
    
    print 'tokenization...'
    tokenizer = PTBTokenizer()
    gts  = tokenizer.tokenize(coco)
    res = tokenizer.tokenize(cocoRes)

    print 'setting up scorers...'
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

    for scorer, method in scorers:
        print 'computing %s score...'%(scorer.method())
        score, scores = scorer.compute_score(gts, res)
        print(score)



def score_numerous_epochs(num_of_ep):
    for i in range(num_of_ep):
        score(i)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch",
    type=int,
    default=0,
    help="number of epoch to test")
parser.add_argument('--dir',
    type = str,
    default='3_res',
    help='Directory of model: ./../../Zaidel/cap_model_[your_input]/')
args = parser.parse_args()
score(args.epoch, './../../Zaidel/cap_model_' + args.dir + '/')