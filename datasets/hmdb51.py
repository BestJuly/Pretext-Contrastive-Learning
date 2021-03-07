"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import accimage
import pdb

hmdb51_dataset_path = 'path/to/hmdb51/jpegs_256'
# e.g. '/my/path/to/dataset/hmdb51/jpegs_256'

def image_to_np(image):
  image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
  image.copyto(image_np)
  image_np = np.transpose(image_np, (1,2,0))
  return image_np


def readim(image_name):
  # read image
  img_data = accimage.Image(image_name)
  img_data = image_to_np(img_data) # RGB
  return img_data


def load_from_frames(foldername, framenames, start_index, tuple_len, clip_len, interval):
  clip_tuple = []
  for i in range(tuple_len):
      one_clip = []
      for j in range(clip_len):
          im_name = os.path.join(foldername, framenames[start_index + i * (tuple_len + interval) + j])
          im_data = readim(im_name)
          one_clip.append(im_data)
      #one_clip_arr = np.array(one_clip)
      clip_tuple.append(one_clip)
  return clip_tuple


def load_one_clip(foldername, framenames, start_index, clip_len):
    one_clip = []
    for i in range(clip_len):
        im_name = os.path.join(foldername, framenames[start_index + i])
        im_data = readim(im_name)
        one_clip.append(im_data)

    return np.array(one_clip)


class HMDB51Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len=16, split='1', train=True, val=False, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'hmdb51', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'hmdb51', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'hmdb51', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split'+ self.split)

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]

        class_idx = self.class_label2idx[videoname[:videoname.find('/')]] - 1 # add - 1 because it is range [1,101] which should be [0, 100]
        '''
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        '''
        # videoname = 'ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # to avoid void folder
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]

        framefolder = os.path.join(hmdb51_dataset_path, vid)

        filenames = ['frame000001.jpg']
        for parent, dirnames, filenames in os.walk(framefolder):
            if 'n_frames' in filenames:
                filenames.remove('n_frames')
            filenames = sorted(filenames)
        framenames = filenames
        length = len(framenames)
        '''
        if length < 16:
            print(vid, length)
            print('\n')
            raise
        '''
        # random select a clip for train
        if self.train:
            clip_start = random.randint(0, length - self.clip_len)
            clip = load_one_clip(framefolder, framenames, clip_start, self.clip_len)
            #clip = videodata[clip_start: clip_start + self.clip_len]

            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)

            return clip, torch.tensor(int(class_idx))
        # sample several clips for test
        else:
            all_clips = []
            all_idx = []
            for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.test_sample_num):
                clip_start = int(i - self.clip_len/2)
                #clip = videodata[clip_start: clip_start + self.clip_len]
                clip = load_one_clip(framefolder, framenames, clip_start, self.clip_len)
                if self.transforms_:
                    trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip 
                    seed = random.random()
                    for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms_(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                    clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                else:
                    clip = torch.tensor(clip)
                all_clips.append(clip)
                all_idx.append(torch.tensor(int(class_idx)))

            return torch.stack(all_clips), torch.tensor(int(class_idx))

class HMDB51ClipRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample clips for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, sample_num, train=True, transforms_=None, split='1'):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'hmdb51', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        self.split = split

        if self.train:
            train_split_path = os.path.join(root_dir, 'hmdb51', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'hmdb51', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index [0-100]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        class_idx = self.class_label2idx[videoname[:videoname.find('/')]]
        '''
        filename = os.path.join(self.root_dir, 'video', videoname)
        videodata = skvideo.io.vread(filename)
        length, height, width, channel = videodata.shape
        '''
        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        # to avoid void folder
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]

        framefolder = os.path.join(hmdb51_dataset_path, vid)

        filenames = ['frame000001.jpg']
        for parent, dirnames, filenames in os.walk(framefolder):
            if 'n_frames' in filenames:
                filenames.remove('n_frames')
            filenames = sorted(filenames)
        framenames = filenames
        length = len(framenames)

        all_clips = []
        all_idx = []
        for i in np.linspace(self.clip_len/2, length-self.clip_len/2, self.sample_num):
            clip_start = int(i - self.clip_len/2)
            #clip = videodata[clip_start: clip_start + self.clip_len]
            clip = load_one_clip(framefolder, framenames, clip_start, self.clip_len)
            if self.transforms_:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
            else:
                clip = torch.tensor(clip)
            all_clips.append(clip)
            all_idx.append(torch.tensor(int(class_idx)))

        return torch.stack(all_clips), torch.stack(all_idx)