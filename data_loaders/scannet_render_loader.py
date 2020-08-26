import os
import glob
import torch
import gzip
import pandas as pd
import numpy as np
#rom torchvision import io
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from base import BaseDataLoader

#from torch.utils.data.dataloader import default_collate

_UV_CHANNELS = 2

class UVDataset(Dataset):
    def __init__(self, uv_color_filenames, compressed_input, transform=None):
        self.transform = transform
        self.uv_color_filenames = uv_color_filenames
        self.compressed_input = compressed_input
        pass

    def __len__(self):
        return len(self.uv_color_filenames)

    def __getitem__(self, idx):
        uv_image_path, color_image_path = self.uv_color_filenames[idx]

        # TODO: Is there s single library to use both for loading images and raw files?
        color_image = io.imread(color_image_path)
        color_image = np.array(color_image)

        image_height, image_width, _ = color_image.shape

        if self.compressed_input:
            # Decompress texture coordinate file into a numpy array
            with gzip.open(uv_image_path, 'rb') as f:
                uv_image = np.frombuffer(f.read(), dtype='float32')
        else:
            uv_image = np.fromfile(uv_image_path, dtype='float32')

        uv_image = np.reshape(uv_image, (image_height, image_width, _UV_CHANNELS))
        # TODO: Try contiguous
        # TODO: Remove copying here. Without it, flipping creates a "negative stride" error, which
        # might just be a problem with PyTorch. The only suggestion online so far is to copy the array
        # but this is likely a huge waste of resources
        uv_image = np.flip(uv_image, axis=0).copy()

        sample = {'uv': uv_image, 'color': color_image}

        if self.transform:
            sample = self.transform(sample)

        # TODO: Don't pack and unlack samples like this.
        # TODO: Don't convert to float here
        return sample['uv'], sample['color'].float()

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

        # TODO: USing Nearest Neighbor Resizing. Bilinear better?
        # TODO: NOTE resize automatically adjusts range of color_image from uint8 [0, 255]
        #  to float64 [0, 1]. Then Rescale converts this to float64 [-1, 1] and __getitem__
        #  to float32 [-1, 1]. This could be a big performance hit. Converting to float32 here
        #  though could mean losing a bit of precision when converting from [0, 1] to [-1, 1].
        #  What is the right way to do this conversion?
        input_image = transform.resize(input_image, (new_h, new_w), order=0)

        # Use bilinear interpolation for color image and anti-aliasing if resize
        #  is downsampling
        color_image = transform.resize(color_image, (new_h, new_w), order=1, anti_aliasing=(h > new_h))

        return {'uv': input_image, 'color': color_image}

class RandomCrop(object):
    def __init__(self, min_crop_scale, max_crop_scale):
        assert min_crop_scale <= max_crop_scale
        self.min_crop_scale = min_crop_scale
        self.max_crop_scale = max_crop_scale

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']
        
        # Assuming input_image and color_image are the same shape
        h, w, c = color_image.shape

        min_size_crop_h = np.round(h * self.min_crop_scale).astype(int)
        max_size_crop_h = np.round(h * self.max_crop_scale).astype(int)

        # Get a crop size while maintaining aspect ratio
        size_crop_h = np.random.randint(min_size_crop_h, max_size_crop_h) if min_size_crop_h < max_size_crop_h else max_size_crop_h
        size_crop_w = np.round(w * size_crop_h / h).astype(int)

        # Get a valid starting and end positions
        h_start = np.random.randint(0, h - size_crop_h) if size_crop_h < h else 0
        w_start = np.random.randint(0, w - size_crop_w) if size_crop_w < w else 0
        h_end = h_start + size_crop_h
        w_end = w_start + size_crop_w
        
        # Crop the input and target
        input_image = input_image[h_start:h_end, w_start:w_end, :]
        color_image = color_image[h_start:h_end, w_start:w_end, :]

        return {'uv': input_image, 'color': color_image}

class Normalize(object):
    """Normalize color images between [-1,1]."""

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']
        # NOTE: Don't normalize input_image. It's just a matrix of coordinates

        # TODO: Move all normalization to this function
        #color_image = (color_image / 127.5) - 1
        color_image = (color_image * 2.0) - 1

        return {'uv': input_image, 'color': color_image}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']

        # NOTE: Axis swapping is not necessary for uv coords since
        #  it is not an image, but rather a matrix of coordinates

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #input_image = input_image.transpose((2, 0, 1))
        color_image = color_image.transpose((2, 0, 1))
        return {'uv': torch.from_numpy(input_image),
                'color': torch.from_numpy(color_image)}


class UVDataLoader(BaseDataLoader):
    # TODO: Rewrite this class in a more understandable way
    def __init__(self, data_dir, uv_folder_name, color_folder_name, data_select_file, batch_size, shuffle, skip,
                 net_input_height, net_input_width, min_crop_scale=1.0, max_crop_scale=1.0, slice_start=None, slice_end=None, compressed_input=False,
                 num_workers=1, training=True):

        # Note on data augmentation
        #  First we crop the original image
        #  Then we resize to input_height, input_width. This is the size of the input to the model

        self.data_dir = data_dir
        self.skip = skip
        self.size = (net_input_height, net_input_width)
        self.min_crop_scale = min_crop_scale
        self.max_crop_scale = max_crop_scale
        #self.min_crop_size = (np.round(input_height * self.crop_scale).astype(int),
        #                  np.round(input_width * self.crop_scale).astype(int))
        self.compressed_input = compressed_input

        with open(os.path.join(data_dir, data_select_file)) as csv_file:
            data = pd.read_csv(csv_file, delimiter=' ', index_col=None, header=None)
            self.use_indices = np.array(data.values).squeeze()

        self.input_color_filenames = self.load_input_color_filenames(data_dir, uv_folder_name, color_folder_name)
        self.input_color_filenames = [self.input_color_filenames[i] for i in self.use_indices if
                                      slice_start <= i < slice_end]
        self.input_color_filenames = self.input_color_filenames[slice(slice_start, slice_end)]

        train_filenames = self.generate_temporal_train_split(self.input_color_filenames, self.skip)
        self.dataset = UVDataset(train_filenames, compressed_input=self.compressed_input, transform=transforms.Compose([
            # TODO: Add data augmentation
            RandomCrop(self.min_crop_scale, self.max_crop_scale),
            Rescale(self.size),
            Normalize(),
            ToTensor()]))

        super().__init__(self.dataset, batch_size, shuffle, num_workers)
        pass

    def split_validation(self):
        val_filenames = self.generate_temporal_val_split(self.input_color_filenames, self.skip)
        val_dataset = UVDataset(val_filenames, compressed_input=self.compressed_input, transform=transforms.Compose([
            Rescale(self.size),
            Normalize(),
            ToTensor()]))

        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def load_input_color_filenames(self, data_dir, uv_folder_name, color_folder_name):
        input_filenames = self.load_filenames_sorted(data_dir, uv_folder_name)
        color_filenames = self.load_filenames_sorted(data_dir, color_folder_name)

        # TODO: Remote 500 sample limit
        input_color_filenames = list(zip(input_filenames, color_filenames))

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
