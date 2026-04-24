import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import hdf5storage
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, hilbert
from utils.DatasetClass import InfiniteDataLoader, SimpleDataset_add
from sklearn.model_selection import train_test_split
from config_arg import load_args

def _norm(data):
    mean, std = data.mean(), data.std()
    data = (data - mean) / std
    return data

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def data_process(args,  x, y, domain_labels):
    def preprocess_data(data):
        data = np.array([_norm(np.squeeze(d)) for d in data])
        data_fft = np.fft.fft(data, axis=-1)
        data_magnitude = np.abs(data_fft)[:, :data_fft.shape[1] // 2]
        return data_magnitude.astype(np.float32)

    print(torch.unique(torch.tensor(y), return_counts=True))

    data_train, data_test, label_train, label_test, domain_train, domain_test = train_test_split(
        x, y, domain_labels, random_state=0, test_size= args.test_size, train_size = args.train_size, stratify=y
    )

    x_train = preprocess_data(data_train)
    x_test = preprocess_data(data_test)

    dataset_train = {
        'data': torch.Tensor(x_train).unsqueeze(1),
        'label': torch.LongTensor(np.array(label_train)),
        'domain_labels': torch.LongTensor(np.array(domain_train))
    }
    dataset_test = {
        'data': torch.Tensor(x_test).unsqueeze(1),
        'label': torch.LongTensor(np.array(label_test)),
        'domain_labels': torch.LongTensor(np.array(domain_test))
    }
    dataset_train, dataset_test = SimpleDataset_add(dataset_train), SimpleDataset_add(dataset_test)

    train_loader = InfiniteDataLoader(dataset_train, batch_size=args.batch_size)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size)

    return train_loader, test_loader

0
class ReadData1:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.domain_label = 0

    def read_data_file(self):
        labels = {'NC Data1': 0, 'IF Data1': 1, 'OF Data1': 1}
        x, y, dlabel = [], [], []
        keycount = 0
        for key in labels.keys():
            file_data = hdf5storage.loadmat(os.path.join(r"your/local/path/to/dataset1/", key))
            file_data = np.array(file_data).reshape(-1, 1)
            win_num = file_data.shape[0] // self.args.data_length
            [x.append(file_data[idx * self.args.data_length:idx * self.args.data_length + self.args.data_length])
             for idx in range(win_num)]
            [y.append(labels[key]) for _ in range(win_num)]
            [dlabel.append(self.domain_label) for _ in range(win_num)]
        train_loader, test_loader = data_process(self.args, x, y, dlabel)
        return train_loader, test_loader

class ReadData2:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.domain_label = 1

    def read_data_file(self):
        labels = {'NC Data2': 0, 'IF Data2': 1, 'OF Data2': 1}
        x, y, dlabel = [], [], []
        keycount = 0
        for key in labels.keys():
            file_data = hdf5storage.loadmat(os.path.join(r"your/local/path/to/dataset2/", key))
            file_data = np.array(file_data).reshape(-1, 1)
            win_num = file_data.shape[0] // self.args.data_length
            [x.append(file_data[idx * self.args.data_length:idx * self.args.data_length + self.args.data_length])
             for idx in range(win_num)]
            [y.append(labels[key]) for _ in range(win_num)]
            [dlabel.append(self.domain_label) for _ in range(win_num)]
        train_loader, test_loader = data_process(self.args, x, y, dlabel)
        return train_loader, test_loader

class ReadData3:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.domain_label = 2

    def read_data_file(self):
        labels = {'NC Data3': 0, 'IF Data3': 1, 'OF Data3': 1}
        x, y, dlabel = [], [], []
        keycount = 0
        for key in labels.keys():
            file_data = hdf5storage.loadmat(os.path.join(r"your/local/path/to/dataset3/", key))
            file_data = np.array(file_data).reshape(-1, 1)
            win_num = file_data.shape[0] // self.args.data_length
            [x.append(file_data[idx * self.args.data_length:idx * self.args.data_length + self.args.data_length])
             for idx in range(win_num)]
            [y.append(labels[key]) for _ in range(win_num)]
            [dlabel.append(self.domain_label) for _ in range(win_num)]
        train_loader, test_loader = data_process(self.args, x, y, dlabel)
        return train_loader, test_loader

class ReadData4:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.domain_label = 3

    def read_data_file(self):
        labels = {'NC Data4': 0, 'IF Data4': 1, 'OF Data4': 1}
        x, y, dlabel = [], [], []
        keycount = 0
        for key in labels.keys():
            file_data = hdf5storage.loadmat(os.path.join(r"your/local/path/to/dataset4/", key))
            file_data = np.array(file_data).reshape(-1, 1)
            win_num = file_data.shape[0] // self.args.data_length
            [x.append(file_data[idx * self.args.data_length:idx * self.args.data_length + self.args.data_length])
             for idx in range(win_num)]
            [y.append(labels[key]) for _ in range(win_num)]
            [dlabel.append(self.domain_label) for _ in range(win_num)]
        train_loader, test_loader = data_process(self.args, x, y, dlabel)
        return train_loader, test_loader


if __name__ == '__main__':
    arg = load_args()
    x, y = ReadData1(arg).read_data_file()
    print('OK')
