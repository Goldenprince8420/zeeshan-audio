import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn


class LJSpeechDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.file_list[idx])
        data = np.load(file_name)

        # Extracting data
        metadata = data["metadata"]
        audio = data["audio"]
        mel_spec = data["mel_spec"]

        # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
        tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
        tensor_meta = torch.tensor(metadata).reshape(-1, 1)
        tensor_mel_spec = torch.tensor(mel_spec)
        return tensor_audio, tensor_meta, tensor_mel_spec


class AudioMNISTDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.file_list[idx])
        data = np.load(file_name)

        # print(self.file_list[idx])
        label, subfolder_index, data_index = self.file_list[idx].split("_")

        # Extracting data
        metadata = data["metadata"]
        audio = data["audio"]
        label = data["label"]
        mel_spec = data["mel_spec"]
        mel_spec_db = data["mel_spec_db"]

        # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
        tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
        tensor_meta = torch.tensor(metadata).reshape(-1, 1)
        tensor_label = torch.nn.functional.one_hot(torch.tensor(int(label)), num_classes=10).reshape(-1, 1)
        tensor_mel_spec = torch.tensor(mel_spec)
        tensor_mel_spec_db = torch.tensor(mel_spec_db)
        return tensor_audio, tensor_meta, tensor_label, tensor_mel_spec, tensor_mel_spec_db


class LibriSpeechDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.file_list[idx])
        data = np.load(file_name)

        # Extracting data
        metadata = data["metadata"]
        audio = data["audio"]
        mel_spec = data["mel_spec"]

        # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
        tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
        tensor_meta = torch.tensor(metadata).reshape(-1, 1)
        tensor_mel_spec = torch.tensor(mel_spec)
        return tensor_audio, tensor_meta, tensor_mel_spec


class VCTKDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.file_list[idx])
        data = np.load(file_name)

        # Extracting data
        metadata = data["metadata"]
        audio = data["audio"]
        mel_spec = data["mel_spec"]

        # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
        tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
        tensor_meta = torch.tensor(metadata).reshape(-1, 1)
        tensor_mel_spec = torch.tensor(mel_spec)
        return tensor_audio, tensor_meta, tensor_mel_spec


def get_dataloader(dataset, batch_size=32, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    # data_path = "../AudioMNIST/preprocessed_data"
    # batch_size = 32
    # shuffle = True
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # audiomnist_dataset = AudioMNISTDataset(data_path=data_path)
    # audiomnist_dataloader = get_dataloader(audiomnist_dataset, batch_size=batch_size, shuffle=shuffle)
    #
    # for i, (real_audio, meta, labels, mel_spec, mel_spec_db) in enumerate(audiomnist_dataloader):
    #     # Move data to device
    #     real_audio = real_audio
    #     labels = labels
    #
    #     print(real_audio.shape)
    #     print(labels.shape)
    #     break

    data_path_ljspeech = "../LJSpeech-1.1/LJSpeech-1.1/processed_data"
    data_path_librispeech = "../train-clean-100/LibriSpeech/processed_data"
    data_path_vctk = "../VCTK/VCTK-Corpus/processed_data"
    batch_size = 32
    shuffle = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ljspeech_dataset = LJSpeechDataset(data_path=data_path_ljspeech)
    ljspeech_dataloader = get_dataloader(ljspeech_dataset, batch_size=batch_size, shuffle=shuffle)

    for i, (real_audio, meta, mel_spec) in enumerate(ljspeech_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.shape)
        print(mel_spec.shape)
        break
    pass

    librispeech_dataset = LibriSpeechDataset(data_path=data_path_librispeech)
    librispeech_dataloader = get_dataloader(librispeech_dataset, batch_size=batch_size, shuffle=shuffle)

    for i, (real_audio, meta, mel_spec) in enumerate(librispeech_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.shape)
        print(mel_spec.shape)
        break
    pass

    vctk_dataset = VCTKDataset(data_path=data_path_vctk)
    vctk_dataloader = get_dataloader(vctk_dataset, batch_size=batch_size, shuffle=shuffle)

    for i, (real_audio, meta, mel_spec) in enumerate(vctk_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.shape)
        print(mel_spec.shape)
        break
    pass
