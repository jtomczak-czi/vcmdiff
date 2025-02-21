import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch_dct as dct

from utils.simple_functions import log1p

class FourierDataset(Dataset):
    """Dataset for training the diffusion.py model on Fourier coefficients."""

    def __init__(self, file_name='data/pancreas.npy'):
        super().__init__()
        self.num_samples = None
        self.N = None
        self.file_name = file_name
        self.data = self.generate_data()

    def generate_data(self):
        """Generate synthetic Fourier coefficient data."""
        data = np.load(self.file_name)
        print('loading data')
        print(data.shape)

        data = torch.tensor(data, dtype=torch.float32)
        self.num_samples = data.shape[0]
        self.N = data.shape[1]
        D = []
        for i in range(self.num_samples):
            signal = data[i]  # Get signal
            fft_coeffs = torch.fft.fft(signal)  # Compute FFT
            # Sort coefficients based on frequency (use fftshift)
            sorted_coeffs = torch.fft.fftshift(fft_coeffs)
            real = fft_coeffs.real.unsqueeze(0)  # Real part
            imag = fft_coeffs.imag.unsqueeze(0)  # Imaginary part
            D.append(torch.cat([real, imag], dim=0))  # Stack as (2, N)
        D = torch.stack(D, dim=0)  # Stack as (num_samples, 2, N)
        return D

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0  # Dummy label


class LoadNumpyData(object):
    def __init__(self, file_name: str='data/pancreas.npy', ratios=(0.7, 0.15, 0.15)):
        super().__init__()
        assert (ratios[0] + ratios[1] + ratios[2]) == 1
        assert (ratios[0] > 0.) and (ratios[1] > 0.) and (ratios[2] > 0.)

        self.data = np.load(file_name)
        print('Data loaded')

        self.N = self.data.shape[0]
        N_train = int(np.round(self.N * ratios[0]))
        N_val = N_train + int(np.round(self.N * ratios[1]))

        self.data_train = self.data[:N_train]
        self.data_val = self.data[N_train:N_val]
        self.data_test = self.data[N_val:]

    def get_data(self, data_type='train'):
        if data_type == 'train':
            return self.data_train
        elif data_type == 'val':
            return self.data_val
        elif data_type == 'test':
            return self.data_test
        else:
            raise ValueError('Wrong data type, it must be in [`train`, `val`, `test`].')


class DCTDataset(Dataset):
    """Dataset for training the diffusion.py model on DCT coefficients."""

    def __init__(self, data_raw, shift=None, scale=None):
        super().__init__()
        self.num_samples = None
        self.N = None
        self.shift = shift
        self.scale = scale
        self.data = self.generate_data(data_raw)

    def generate_data(self, data_raw):
        """Generate DCT"""
        print(f'Loaded data of the following shape: {data_raw.shape}')

        data = torch.tensor(data_raw, dtype=torch.float32)
        self.num_samples = data.shape[0]
        self.N = data.shape[1]
        # iterate over datapoints
        D = []
        for i in range(self.num_samples):
            signal = data[[i]]
            signal_dct = dct.dct(signal, norm='ortho')
            D.append(signal_dct)  # Stack as (C, D)
        D = torch.stack(D, dim=0)  # Stack as (num_samples, C, D)

        # Scale: data = data / max(abs(data))
        if self.shift is None:
            self.shift = torch.mean(D, dim=0, keepdim=True)

        if self.scale is None:
            # self.scale, _ = torch.max(torch.abs(D), dim=0, keepdim=True)
            self.scale = torch.std(D, dim=0, keepdim=True)

        D = (D - self.shift) / self.scale
        return D

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0  # Dummy label


class NormalizedDataset(Dataset):
    """Dataset for training the diffusion.py model on Fourier coefficients."""

    def __init__(self, data_raw, normalize_counts=True, shift=None, scale=None):
        super().__init__()
        self.EPS = 1.e-5
        self.num_samples = None
        self.D = None
        self.normalize_counts = normalize_counts
        self.shift = shift
        self.scale = scale
        self.data = self.generate_data(data_raw)

    def generate_data(self, data_raw):
        """Generate DCT"""
        print(f'Loaded data of the following shape: {data_raw.shape}')

        data = torch.tensor(data_raw, dtype=torch.float32)
        self.num_samples = data.shape[0]
        self.D = data.shape[1]

        # counts -> simplex
        if self.normalize_counts:
            data = data / data.sum(1, keepdim=True)

        # simplex -> hypercube [0,1]
        # Scale: data = data / max(abs(data))
        if self.scale is None:
            self.scale, _ = torch.max(data, dim=0, keepdim=True)
            # self.scale = torch.std(data, dim=0, keepdim=True)

        # normalize
        data = (data) / (self.scale + self.EPS)
        # clamp just in case!
        data = torch.clamp(data.unsqueeze(1), 0., 1.)
        # hypercube [0,1] -> hypercube [-1,+1]
        data = 2 * data - 1.

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0  # Dummy label


class CountDataset(Dataset):
    """Dataset for training the diffusion.py model on Fourier coefficients."""

    def __init__(self, data_raw):
        super().__init__()
        self.EPS = 1.e-5
        self.num_samples = None
        self.D = None
        self.data = self.generate_data(data_raw)

    def generate_data(self, data_raw):
        """Prepare Counts"""
        print(f'Loaded data of the following shape: {data_raw.shape}')

        data = torch.tensor(data_raw, dtype=torch.float32)
        self.num_samples = data.shape[0]
        self.D = data.shape[1]

        return data.unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], 0  # Dummy label


class CountOccurenceDataset(Dataset):
    """Dataset for training the diffusion.py model on Fourier coefficients."""

    def __init__(self, data_raw):
        super().__init__()
        self.EPS = 1.e-5
        self.num_samples = None
        self.D = None
        self.gene_count, self.gene_occurence = self.generate_data(data_raw)

    def generate_data(self, data_raw):
        """Generate DCT"""
        print(f'Loaded data of the following shape: {data_raw.shape}')

        gene_count = torch.tensor(data_raw, dtype=torch.float32)
        gene_occurence = torch.tensor(torch.gt(gene_count, 0.), dtype=torch.float32)

        self.num_samples = gene_count.shape[0]
        self.D = gene_count.shape[1]

        return gene_count.unsqueeze(1), gene_occurence.unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.gene_count[idx], self.gene_occurence[idx]  # Dummy label