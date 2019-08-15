# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
import math
from scipy import ndimage   
import torchvision.transforms as transforms

class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, txt_path, vid_path, vid_pad, txt_pad, kind):
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.txt_path = txt_path
        self.vid_path = vid_path
        self.kind = kind
        f = open(self.txt_path, 'r')
        lines = f.readlines()
        print(len(lines))
        # self.data = []
        
        
        line_lists = [line.strip().split(',') for line in lines]
            # print(line_list)
            # if line_list[0] in os.listdir(self.vid_path):
        self.data = [(line_list[0], line_list[2], int(float(line_list[3])*25)+1, int(float(line_list[4])*25)+1) 
                        for line_list in line_lists]
        self.data = list(filter(lambda data: int(int(data[3])-int(data[2])) <= self.vid_pad, self.data))
        print(len(self.data))
        # print(os.listdir(self.vid_path))
        # print(self.data)
        # self.videos = glob.glob(os.path.join(vid_path, "*"))
        # print(self.videos)
        # self.videos = list(filter(lambda dir: len(os.listdir(dir)) == 29, self.videos))
        # print(self.videos)
        # self.data = []
        # for vid in self.videos:
        #     items = vid.split(os.path.sep)
        #     # print(items, items[-2], items[-1])            
        #     self.data.append((vid, items[-3], items[-2]))        
        # print(self.data)
        # print(self.data)       
    def __getitem__(self, idx):
        # self.data = list(filter(lambda x: x[0] in os.listdir(self.vid_path), self.data))
        # print(self.data[0])
        # print(idx)
        (vid_folder, anno_txt, st_time, ed_time) = self.data[idx]
        # print(anno_txt)
        # print(vid_folder, st_time, ed_time)
        vid = self._load_vid(vid_folder, st_time, ed_time)
        anno_txt = anno_txt.upper()
        anno = self._load_anno(anno_txt)
        # anno_len = anno.shape[0]
        # vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        # if(self.kind == 'train'):
        #     vid = HorizontalFlip(vid)
        #     vid = FrameRemoval(vid)
        # vid = ColorNormalize(vid)
        # print('vid.transpose:', vid.transpose(3,0,1,2).shape)
        # print(anno_txt, anno)
        inputs = torch.FloatTensor(vid)
        labels = torch.LongTensor(anno)
        # print(labels.size())
        return {'encoder_tensor': inputs, 'decoder_tensor': labels}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, folder, st, ed): 
        # files = sorted(os.listdir(p)) 
        p = os.path.join(self.vid_path, folder) 
        # print(p)
        # dirs = os.listdir(p)
        # print(dirs)
        # for dir in dirs:
        # if len(os.listdir(p)) == 0:
        #         print(p)
        # files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))
        # s = int((float(st)*25))+1
        # e = int((float(ed)*25))+1
        # for file in files[s:e]:
        #     print(os.path.join(p, file))
        files =  [os.path.join(p, '{}.jpg'.format(i)) for i in range(st, ed)]
        files = list(filter(lambda path:os.path.exists(path), files))
        array = [ndimage.imread(file) for file in files]

        # array = [cv2.cvtColor(file, cv2.COLOR_RGBA2BGR) for file in array]
        # array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        # array = [cv2.resize(im, (112, 112)) for im in array]
        array = bbc(array, self.vid_pad, True)
        # print(len(array))
        # print(array)
        # array = np.stack(array, axis=0)
        # print(array.shape)
        return array

    
    def _load_anno(self, name):
        return MyDataset.txt2arr(name, 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, SOS=False):
        # SOS: 1, EOS: 2, P: 0, OTH: 3+x
        arr = []
        if(SOS):            
            tensor = [1]
        else:
            tensor = []
        for c in list(txt):
            tensor.append(3 + MyDataset.letters.index(c))
        tensor.append(2)
        return np.array(tensor)
    
    @staticmethod
    def tensor2text(tensor):
        # (B, T)
        result = []
        n = tensor.size(0)
        T = tensor.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = tensor[i,t]
                if(c == 2): break
                elif(c == 3): text.append(' ')
                elif(3 < c): text.append(chr(c-4+ord('a')))
            text = ''.join(text)
            result.append(text)
        return result

    @staticmethod
    def arr2txt(arr):       
        # (B, T)
        result = []
        n = arr.size(0)
        T = arr.size(1)
        for i in range(n):
            text = []
            for t in range(T):
                c = arr[i,t]
                if(c >= 3):
                    text.append((MyDataset.letters[c - 3]).lower())
            text = ''.join(text)
            result.append(text)
        return result
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt)
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()        

    @staticmethod
    def ED(predict, truth):
        ED = [1.0*editdistance.eval(p[0], p[1]) for p in zip(predict, truth)] 
        return ED

# print(batch['encoder_tensor'], batch['decoder_tensor'])
# import options as opt
# from torch.utils.data import Dataset, DataLoader

# data = MyDataset(opt.trn_txt_path, opt.vid_path, 
#             opt.vid_pad,
#             opt.txt_pad, kind='train')
# # print(len(data))
# loader = DataLoader(data, 
#         batch_size=opt.batch_size,
#         num_workers=opt.num_workers,
#         drop_last=False,
#         shuffle=True)  
   
# for idx, batch in enumerate(loader):
#     print(batch['encoder_tensor'].size())
    # print(len(batch['encoder_tensor']))    
# datafile = data[0]
# print(datafile['encoder_tensor'].shape)