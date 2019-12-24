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

        if self._is_for == "train":
            self._data_folder = opt[self._name]["train_folder"]
            #self._data_file = opt[self._name]["train_file"]
        elif self._is_for == "val":
            self._data_folder = opt[self._name]["val_folder"]
            #self._data_file = opt[self._name]["val_file"]
        elif self._is_for == "test":
            self._data_folder = opt[self._name]["test_folder"]
            #self._data_file = opt[self._name]["test_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # get data
        imgs, target = self._data[index], self._targets[index]

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

        data_filenames = [f for f in listdir(os.path.join(self._root, self._data_folder)) if os.path.isfile(os.path.join(self._root, self._data_folder, f))]

        self._data = []
        self._targets = []

        id = 0

        for current_filename in data_filenames:

            data_filepath = os.path.join(self._root, self._data_folder, current_filename)
            id = id + 1
            print(str(id)+' '+data_filepath)

            self._raw_data = np.load(data_filepath, allow_pickle=True)

            # Generate balanced pairs for each file and concatenate classes

            self.generate_balanced_pairs(self._raw_data)

        # Dataset size
        self._dataset_size = self._data.__len__()

    def generate_balanced_pairs(self, raw_data):

        num_of_classes = raw_data.shape[0]
        cte = 20

        for i in range(num_of_classes*cte):

            # Select 2 different classes randomly (rnd_cls1 and rnd_cls2)

            rnd_cls1, rnd_cls2 = random.sample(range(0, num_of_classes), 2)

            # Select 2 samples of rnd_cls1 and one of a rnd_cls2

            # Select rnd_cls1 subclass randomly

            num_of_subclasses = raw_data[rnd_cls1][1].__len__()
            rnd_subcls = random.sample(range(0, num_of_subclasses), 1)[0]

            # Select two random samples from the random subclass
            num_of_subsamples = raw_data[rnd_cls1][1][rnd_subcls].shape[0]
            rnd_subsmpl1, rnd_subsmpl2 = random.sample(range(0, num_of_subsamples), 2)

            # Select rnd_cls2 subclass randomly
            num_of_subclasses = raw_data[rnd_cls2][1].__len__()
            rnd_subcls2 = random.sample(range(0, num_of_subclasses), 1)[0]

            # Select one random sample from the random subclass
            num_of_subsamples = raw_data[rnd_cls2][1][rnd_subcls2].shape[0]
            rnd_subsmpl3 = random.sample(range(0, num_of_subsamples), 1)[0]

            data_sample1 = raw_data[rnd_cls1][1][rnd_subcls][rnd_subsmpl1]
            data_sample2 = raw_data[rnd_cls1][1][rnd_subcls][rnd_subsmpl2]
            data_sample3 = raw_data[rnd_cls2][1][rnd_subcls2][rnd_subsmpl3]

            data_sample1 = self.normalize_in_range([0, 1], data_sample1)
            data_sample2 = self.normalize_in_range([0, 1], data_sample2)
            data_sample3 = self.normalize_in_range([0, 1], data_sample3)


            self._data.append([data_sample1, data_sample2, data_sample3])
            self._targets.append([1, 0])

            # if 'true':
            #     font1 = {'color': 'green'}
            #     font2 = {'color': 'darkred'}
            #
            #     plt.plot([1,2,3])
            #     plt.subplot(1, 3, 1)
            #     plt.imshow(data_sample1, cmap='Greys_r')
            #     plt.title('Class {}\n Subclass {}\n sample {}'.format(rnd_cls1, rnd_subcls, rnd_subsmpl1), fontdict=font1)
            #     plt.subplot(1, 3, 2)
            #     plt.imshow(data_sample2, cmap='Greys_r')
            #     plt.title('Class {}\n Subclass {}\n sample {}'.format(rnd_cls1, rnd_subcls, rnd_subsmpl2), fontdict=font1)
            #     plt.subplot(1, 3, 3)
            #     plt.imshow(data_sample3, cmap='Greys_r')
            #     plt.title('Class {}\n Subclass {}\n sample {}'.format(rnd_cls2, rnd_subcls2, rnd_subsmpl3), fontdict=font2)
            #     plt.show()

    def normalize_in_range(self, norm_range, raw_array):

        # Normalize
        max = np.amax(raw_array)
        min = np.amin(raw_array)
        norm_array = (norm_range[1] - norm_range[0]) / (max - min) * (raw_array - max) + norm_range[1]

        return norm_array
