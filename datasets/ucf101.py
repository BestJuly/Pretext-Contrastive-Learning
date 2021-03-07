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


ucf101_dataset_path = 'path/to/ucf101/jpegs_256'
# e.g. '/my/path/to/dataset/ucf101/jpegs_256'

def spatial_rorate(x):
    # (C X T x H x W)
    #return torch.rot90(x, random.randint(0,2) + 1, [2, 3])
    return torch.rot90(x, 1, [2, 3])

def adjacent_shuffle(x):
    # (C X T x H x W)
    tmp = torch.chunk(x, 4, dim=1)
    order = [1,2,3,4]
    ind1 = random.randint(0,3)
    ind2 = (ind1 + random.randint(0,2) + 1) % 4
    order[ind1], order[ind2] = order[ind2], order[ind1]
    x_new = torch.cat((tmp[order[0]], tmp[order[1]], tmp[order[2]], tmp[order[3]]),1)
    return x_new

def spatial_permutation(x):
    c, t, h, w = x.shape
    hm = h // 2
    wm = w // 2
    slices = []
    slices.append(x[:,:,:hm,:wm]) # A
    slices.append(x[:,:,:hm,wm:]) # B
    slices.append(x[:,:,hm:,:wm]) # C
    slices.append(x[:,:,hm,wm:]) # D
    order = [1,2,3,4]
    while order == [1,2,3,4]:
        random.shuffle(order)
    #order = [4,2,3,1]

    x_new = torch.cat(torch.cat((slices[order[0]], slices[order[1]]), 3), torch.cat((slices[order[3]], slices[order[4]]), 3), 2)
    return x_new


def options_func(x, label):
    #options = [0, 1, 2, 3, 4] # origin, rotation, spatial permtation, temporal shuffling, remote clip
    funcs = [spatial_rorate, spatial_permutation, adjacent_shuffle]
    func = funcs[label - 1]
    return func(x)


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


def load4clips(foldername, framenames, start_index, tuple_len, clip_len, interval):
    label = random.randint(0,4) # 0: origin, 1: rotation, 2: spatial permtation, 3: temporal shuffling, 4: remote clip
    # use temporal information only

    dist = 16 # remote distance
    clip_tuple = []
    for i in range(3):
        clip_tuple.append(load_one_clip(foldername, framenames, start_index + i * (tuple_len + interval), clip_len))
    if label != 4:
        clip_tuple.append(load_one_clip(foldername, framenames, start_index + 1 * (tuple_len + interval), clip_len))
    else:
        # remote_flag = True
        length = len(framenames)
        tuple_forward = start_index - clip_len - dist
        tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)
        tuple_backward = length - start_index - tuple_total_frames - clip_len - dist
        if tuple_forward < 0 or tuple_backward < 0:
            # remote_flag = False
            clip_tuple.append(load_one_clip(foldername, framenames, start_index + 1 * (tuple_len + interval), clip_len))
            label = random.randint(0,3)
        else:
            if tuple_forward < 0 and tuple_backward > 0:
                clip_start = random.randint(start_index + tuple_total_frames + dist, length - clip_len)
            elif tuple_forward > 0 and tuple_backward < 0:
                clip_start = random.randint(0, start_index - clip_len - dist)
            else:
                clip_starts = [random.randint(0, start_index - clip_len - dist), random.randint(start_index + tuple_total_frames + dist, length - clip_len)]
                clip_start = clip_starts[random.randint(0,1)]
            remote_clip = load_one_clip(foldername, framenames, clip_start, clip_len)
            clip_tuple.append(remote_clip)
    return clip_tuple, label


def load_one_clip(foldername, framenames, start_index, clip_len):
    one_clip = []
    for i in range(clip_len):
        im_name = os.path.join(foldername, framenames[start_index + i])
        im_data = readim(im_name)
        one_clip.append(im_data)
    return np.array(one_clip)


class UCF101Dataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """
    def __init__(self, root_dir, clip_len, split='1', train=True, val=False, transforms_=None, test_sample_num=10):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.split = split
        self.train = train
        self.val = val
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'ucf101', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'ucf101', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'ucf101', 'testlist0' + self.split + '.txt')
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

        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]

        # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]

        framefolder = os.path.join(ucf101_dataset_path, vid)
        for parent, dirnames, filenames in os.walk(framefolder):
            if 'n_frames' in filenames:
                filenames.remove('n_frames')
            filenames = sorted(filenames)
        framenames = filenames
        length = len(framenames)

        # random select a clip for train
        if self.train or self.val:
            clip_start = random.randint(0, length - self.clip_len)
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


class UCF101ClipRetrievalDataset(Dataset):
    """UCF101 dataset for Retrieval. Sample clips for each video. The class index start from 0.
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        sample_num(int): number of clips per video.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, sample_num, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.sample_num = sample_num
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        class_idx_path = os.path.join(root_dir, 'ucf101', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]

        if self.train:
            train_split_path = os.path.join(root_dir, 'ucf101', 'trainlist01.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root_dir, 'ucf101', 'testlist01.txt')
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
        #'''
        # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
        #'''

        framefolder = os.path.join(ucf101_dataset_path, vid)
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

class UCF101PCLDataset(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists. 
    
    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len, interval, tuple_len, train=True, transforms_=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        if self.train:
            vcop_train_split_name = 'vcop_train_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_train_split_path = os.path.join(root_dir, 'ucf101', vcop_train_split_name)
            self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        else:
            vcop_test_split_name = 'vcop_test_{}_{}_{}.txt'.format(clip_len, interval, tuple_len)
            vcop_test_split_path = os.path.join(root_dir, 'ucf101', vcop_test_split_name)
            self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

    def __len__(self):
        if self.train:
            return len(self.train_split)
        else:
            return len(self.test_split)

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.train:
            videoname = self.train_split[idx]
        else:
            videoname = self.test_split[idx]
        
        vid = videoname.split(' ')[0]
        vid = vid[:-4].split('/')[1]
        #'''
        # to avoid void folder because different names: HandStandPushups vs HandstandPushups
        vids = vid.split('_')
        if vids[1] == 'HandStandPushups':
            vid = vids[0] + '_HandstandPushups_' + vids[2] + '_' + vids[3]
        #'''

        framefolder = os.path.join(ucf101_dataset_path, vid)
        for parent, dirnames, filenames in os.walk(framefolder):
            if 'n_frames' in filenames:
                filenames.remove('n_frames')
            filenames = sorted(filenames)
        framenames = filenames
        length = len(framenames)

        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        
        clip_start = tuple_start
        tuple_clip, label = load4clips(framefolder, framenames, tuple_start, self.tuple_len, self.clip_len, self.interval)

        #options = [0, 1, 2, 3, 4] # origin, rotation, spatial permtation, temporal shuffling, remote clip

        if self.transforms_:
            trans_tuple = []
            for i in range(4):
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip 
                seed = random.random()
                clip = tuple_clip[i]
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame) # PIL image, this is activated when using skvideo.io
                    frame = self.transforms_(frame) # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
                
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(label), idx 