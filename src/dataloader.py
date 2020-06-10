import os
import glob
import torch
import numpy as np
#rom torchvision import io
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#from torch.utils.data.dataloader import default_collate

_SCREEN_HEIGHT = 968
_SCREEN_WIDTH = 1296
_UV_CHANNELS = 2

_INPUT_SIZE = 256 # Size of input data to the network

class UVDataset(Dataset):
    def __init__(self, uv_color_filenames, transform=None):
        self.transform = transform
        self.uv_color_filenames = uv_color_filenames
        pass

    def __len__(self):
        return len(self.uv_color_filenames)

    def __getitem__(self, idx):
        uv_image_path, color_image_path = self.uv_color_filenames[idx]

        # TODO: Is there s single library to use both for loading images and raw files?
        color_image = io.imread(color_image_path)
        color_image = np.array(color_image)
        uv_image = np.fromfile(uv_image_path, dtype='float32')
        uv_image = np.reshape(uv_image, (_SCREEN_HEIGHT, _SCREEN_WIDTH, _UV_CHANNELS))
        # TODO: Remove copying here. Without it, flipping creates a "negative stride" error, which
        # might just be a problem with PyTorch. The only suggestion online so far is to copy the array
        # but this is likely a huge waste of resources
        uv_image = np.flip(uv_image, axis=0).copy()

        sample = {'uv': uv_image, 'color': color_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']

        h, w = color_image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # Nearest Neighbor Resizing
        input_image = transform.resize(input_image, (new_h, new_w), order=0)
        color_image = transform.resize(color_image, (new_h, new_w), order=0)

        return {'uv': input_image, 'color': color_image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']

        # TODO: Is axis swapping necessary for uv coords?

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_image = input_image.transpose((2, 0, 1))
        color_image = color_image.transpose((2, 0, 1))
        return {'uv': torch.from_numpy(input_image),
                'color': torch.from_numpy(color_image)}


class UVDataLoader(DataLoader):
    # TODO: Rewrite this class in a more understandable way
    def __init__(self, data_dir, batch_size, shuffle, num_workers, skip, training=True):
        self.data_dir = data_dir

        input_color_filenames = self.load_input_color_filenames(data_dir)
        if training:
            train_filenames = self.generate_temporal_train_split(input_color_filenames, skip)
            self.dataset = UVDataset(train_filenames, transform=transforms.Compose([
                Rescale(_INPUT_SIZE),
                # TODO: Add data augmentation
                ToTensor()]))
        else:
            val_filenames = self.generate_temporal_val_split(input_color_filenames, skip)
            self.dataset = UVDataset(val_filenames, transform=transforms.Compose([
                Rescale(_INPUT_SIZE),
                ToTensor()]))


        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            #'collate_fn': default_collate,
            #'sampler': sampler,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)
        pass

    def load_input_color_filenames(self, data_dir):
        input_filenames = self.load_filenames_sorted(data_dir, 'uv')
        color_filenames = self.load_filenames_sorted(data_dir, 'color')

        # TODO: Remote 500 sample limit
        input_color_filenames = list(zip(input_filenames, color_filenames))[0:500]

        return input_color_filenames

    def generate_temporal_train_split(self, temporal_file_pairs, skip):
        num_pairs = len(temporal_file_pairs)

        # Separate train and test by selecting evenly spaced samples throughout the dataset
        train_filenames = [temporal_file_pairs[i] for i in range(num_pairs) if (i % skip) != 0]

        return train_filenames

    def generate_temporal_val_split(self, temporal_file_pairs, skip):
        num_pairs = len(temporal_file_pairs)

        # Separate train and test by selecting evenly spaced samples throughout the dataset
        val_filenames = [temporal_file_pairs[i] for i in range(num_pairs) if (i % skip) == 0]

        return val_filenames

    def load_filenames_sorted(self, data_dir, data_name):
        filenames = glob.glob(os.path.join(data_dir, data_name, '*'))
        filenames.sort()

        return filenames
