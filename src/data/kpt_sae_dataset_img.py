import os.path
from src.data.dataset import DatasetBase
import numpy as np
from torchvision.datasets.utils import download_url
import tarfile
import sys
import pickle
import struct
import random
import matplotlib.pyplot as plt
from os import listdir

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image


class eventSurfacesDataset(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(eventSurfacesDataset, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'event-surfaces-dataset'

        # init meta
        self._init_meta(opt)

        # read dataset
        self._read_dataset()

    def _init_meta(self, opt):
        #self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]
        self._sae_type = opt[self._name]["sae_type"]

        if self._is_for == "train":
            #self._data_folder = opt[self._name]["train_folder"]
            self._data_file = opt[self._name]["train_file"]
        elif self._is_for == "val":
            self._data_file = opt[self._name]["val_file"]
        elif self._is_for == "test":
            self._data_file = opt[self._name]["test_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # Generate balanced pairs for each file and concatenate classes
        current_stream_num = random.sample(range(0, self._num_of_streams), 1)[0]  # Select randomly one of the streams

        # get data
        imgs, target = self.generate_balanced_pairs(current_stream_num)

        # pack data
        sample = {'img1': imgs[0], 'img2': imgs[1], 'img3': imgs[2], 'target1': target[0], 'target2': target[1]}

        # apply transformations
        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):

        # Read all data files of the correspondent folder
        self.data_filenames = [line.rstrip() for line in open(self._data_file)]
        self._num_of_streams = self.data_filenames.__len__()

        self._kpt_names = []
        self._kpt_num = []

        self._data = []
        self._targets = []
        self._mask = []
        self._dataset_size = 0

        arbitrary_factor = self._opt['dataset']['mining_factor']
        id = 0

        for current_filename in self.data_filenames:

            data_filepath = os.path.join(self._root, self._data_file, current_filename)
            id = id + 1
            print(str(id)+' '+current_filename)

            kpts_name_list = os.listdir(os.path.join(self._root, current_filename, self._sae_type))
            self._kpt_names.append(kpts_name_list)
            self._kpt_num.append(kpts_name_list.__len__())
            self._dataset_size = self._dataset_size + kpts_name_list.__len__()

        self._dataset_size = self._dataset_size * arbitrary_factor

    def generate_balanced_pairs(self, current_stream_num):

        num_of_classes = self._kpt_num[current_stream_num]

        # Select 2 different classes randomly (rnd_cls1 and rnd_cls2)
        if num_of_classes == 0:
            print('Error: num_of_classes = 0')

        only_selected_classes = self._opt['dataset']['only_selected_classes']

        if only_selected_classes:  # Predifined_Classes
            # classes_list = [2, 10, 15]
            # rnd_cls1, rnd_cls2 = random.sample(classes_list, 2)
            rnd_cls1 = random.sample(range(0, num_of_classes), 1)[0]
            # Load mask_data
            #valid_kpts = mask_data[:, :, 0]
            #classes_list = np.where(valid_kpts[rnd_cls1] <= 200)[0].tolist()
            #rnd_cls2 = random.sample(classes_list, 1)[0]
        else:
            if num_of_classes < 2:
                print('error_sample')
            rnd_cls1, rnd_cls2 = random.sample(range(0, num_of_classes), 2)

        # Select 2 samples of rnd_cls1 and one of a rnd_cls2

        anchor_class_path = os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls1])
        negative_class_path = os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type,self._kpt_names[current_stream_num][rnd_cls2])

        # Select rnd_cls1 subclass randomly
        subclasses_file_a = np.load(os.path.join(anchor_class_path, 'subclasses.npy'), allow_pickle=True)
        num_of_subclasses_a = subclasses_file_a.shape[0]
        rnd_subcls = random.sample(range(0, num_of_subclasses_a), 1)[0]

        # Select two random samples from the random subclass
        num_of_subsamples_a = subclasses_file_a[rnd_subcls].shape[0]
        rnd_subsmpl1, rnd_subsmpl2 = random.sample(range(0, num_of_subsamples_a), 2)
        samples_list_a = [line.rstrip() for line in open(os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls1],'kpt_files.txt'))]
        #samples_list_a = glob.glob(os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls1],'*.npy'))

        # Select rnd_cls2 subclass randomly
        subclasses_sample_n = np.load(os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls2], 'subclasses.npy'),allow_pickle=True)
        num_of_subclasses_n = subclasses_sample_n.shape[0]
        rnd_subcls2 = random.sample(range(0, num_of_subclasses_n), 1)[0]

        # Select one random sample from the random subclass
        num_of_subsamples_n = subclasses_sample_n[rnd_subcls2].shape[0]
        rnd_subsmpl3 = random.sample(range(0, num_of_subsamples_n), 1)[0]
        samples_list_n = [line.rstrip() for line in open(os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls2], 'kpt_files.txt'))]
        #samples_list_n = glob.glob(os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls2], '*.npy'))


        img_name_1 = os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls1], samples_list_a[rnd_subsmpl1])
        img_name_2 = os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls1], samples_list_a[rnd_subsmpl2])
        img_name_3 = os.path.join(self._root, self.data_filenames[current_stream_num], self._sae_type, self._kpt_names[current_stream_num][rnd_cls2], samples_list_n[rnd_subsmpl3])

        # img_1 = Image.open(img_name_1)
        # img_2 = Image.open(img_name_2)
        # img_3 = Image.open(img_name_3)

        # data_sample1 = np.asarray(img_1)
        # data_sample2 = np.asarray(img_2)
        # data_sample3 = np.asarray(img_3)

        data_sample1 = np.load(img_name_1)
        data_sample2 = np.load(img_name_2)
        data_sample3 = np.load(img_name_3)

        imgs = [data_sample1, data_sample2, data_sample3]
        targets = [1, 0]

        # if 'true':
        #      font1 = {'color': 'green'}
        #      font2 = {'color': 'darkred'}
        #
        #      plt.plot([1,2,3])
        #      plt.subplot(1, 3, 1)
        #      plt.imshow(data_sample1, cmap='Greys_r')
        #      plt.title('Class {}\n sample {}'.format(self._kpt_names[current_stream_num][rnd_cls1], samples_list_a[rnd_subsmpl1][-10:]), fontdict=font1)
        #      #plt.title('Class {}\n Subclass {}\n sample {}'.format(rnd_cls1, rnd_subcls, rnd_subsmpl1), fontdict=font1)
        #      plt.subplot(1, 3, 2)
        #      plt.imshow(data_sample2, cmap='Greys_r')
        #      plt.title('Class {}\n sample {}'.format(self._kpt_names[current_stream_num][rnd_cls1],samples_list_a[rnd_subsmpl2][-10:]), fontdict=font1)
        #      #plt.title('Class {}\n Subclass {}\n sample {}'.format(rnd_cls1, rnd_subcls, rnd_subsmpl2), fontdict=font1)
        #      plt.subplot(1, 3, 3)
        #      plt.imshow(data_sample3, cmap='Greys_r')
        #      plt.title('Class {}\n sample {}'.format(self._kpt_names[current_stream_num][rnd_cls2],samples_list_a[rnd_subsmpl3][-10:]), fontdict=font2)
        #      #plt.title('Class {}\n Subclass {}\n sample {}'.format(rnd_cls2, rnd_subcls2, rnd_subsmpl3), fontdict=font2)
        #      plt.show()

        return imgs, targets

