import os.path
from src.data.dataset import DatasetBase
import numpy as np
from torchvision.datasets.utils import download_url
import tarfile
import sys
import pickle
import struct

class nmnistDataset(DatasetBase):
    def __init__(self, opt, is_for, subset, transform, dataset_type):
        super(nmnistDataset, self).__init__(opt, is_for, subset, transform, dataset_type)
        self._name = 'n-mnist'

        # init meta
        self._init_meta(opt)

        # read dataset
        self._read_dataset()

    def _init_meta(self, opt):
        #self._rgb = not opt[self._name]["use_bgr"]
        self._root = opt[self._name]["data_dir"]

        if self._is_for == "train":
            self._data_folder = opt[self._name]["train_folder"]
            #self._ids_filename = self._opt[self._name]["train_ids_file"]
        elif self._is_for == "val":
            self._data_folder = opt[self._name]["val_folder"]
            #self._ids_filename = self._opt[self._name]["val_ids_file"]
        elif self._is_for == "test":
            self._data_folder = opt[self._name]["test_folder"]
            #self._ids_filename = self._opt[self._name]["test_ids_file"]
        else:
            raise ValueError(f"is_for={self._is_for} not valid")

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # get data
        x, y, pol, ts, target = self._x_data[index], self._y_data[index], self._pol_data[index], self._ts_data[index], self._label_data[index]
        #img, target = self._data[index], self._targets[index]

        # pack data
        sample = {'x': x, 'y': y, 'pol': pol, 'ts': ts, 'target': target}
        #sample = {'img': img, 'target': target}

        # apply transformations
        if self._transform is not None:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):

        data_filepath = os.path.join(self._root, self._data_folder)
        filenames = os.listdir(data_filepath)  # get all files' and folders' names in the current directory

        self._x_data = []
        self._y_data = []
        self._pol_data = []
        self._ts_data = []
        self._label_data = []


        # check whether the current object is a folder or not
        self.class_names = [] # Folder name corresponds to class name
        for filename in filenames:  # loop through all the files and folders
            if os.path.isdir(os.path.join(os.path.abspath(data_filepath), filename)):
                self.class_names.append(filename)


        # for each class read all files

        for class_name in self.class_names:
            class_filepath = os.path.join(data_filepath, class_name)

            bin_names = os.listdir(class_filepath)  # get all files' and folders' names in the current directory

            for bin_name in bin_names:
                file_length_in_bytes = os.path.getsize(os.path.join(class_filepath, bin_name))
                with open(os.path.join(class_filepath, bin_name), mode='rb') as file:  # b is important -> binary
                    fileContent = file.read()
                    #print(fileContent)
                    evtStream = []
                    pol = []
                    ts = []
                    count_line = 0
                    for line in fileContent:
                        evtStream.append(line)
                        if (count_line % 5) == 2:
                            pol_byte = bin(line >> 7)
                            pol.append(int(pol_byte, 2))
                            ts_aux1 = int(bin((line & 127) << 16), 2) #TD.ts = bitshift(bitand(evtStream(3:5: end), 127), 16); % time in microseconds
                        if (count_line % 5) == 3:
                            ts_aux2 = ts_aux1+int(bin(line << 8), 2) #TD.ts = TD.ts + bitshift(evtStream(4:5: end), 8);
                        if (count_line % 5) == 4:
                                ts_aux3 = ts_aux2 + line #TD.ts = TD.ts + evtStream(5:5: end);
                                ts.append(ts_aux3)
                        count_line = count_line+1
                x = evtStream[0:len(evtStream):5]
                y = evtStream[1:len(evtStream):5]
                self._x_data.append(x)
                self._y_data.append(y)
                self._pol_data.append(pol)
                self._ts_data.append(ts)
                self._label_data.append(class_name)


        # reshape data
        #self._data = np.vstack(self._data).reshape(-1, 3, 32, 32)
        #self._data = self._data.transpose((0, 2, 3, 1))  # convert to HWC

        # dataset size
        self._dataset_size = len(self._x_data)






