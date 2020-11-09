import os
import glob
import torch
import gzip
import pandas as pd
import numpy as np
#rom torchvision import io
from skimage import io, transform, img_as_float32
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

        # Stride becomes negative without a copy
        # TODO: Remove need to flip image by optimizing data preprocessing
        uv_image = np.flip(uv_image, axis=0).copy()

        sample = {'uv': uv_image, 'color': color_image}

        if self.transform:
            sample = self.transform(sample)

        # TODO: Don't pack and unlack samples like this.
        return sample['uv'], sample['color']

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, min_size, max_size):
        # For now size is defined as the smaller size of an image
        assert isinstance(min_size, int)
        assert isinstance(max_size, int)
        assert min_size <= max_size
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']

        h, w = color_image.shape[:2]

        output_size = np.random.randint(self.min_size, self.max_size + 1)

        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size

        new_h, new_w = int(new_h), int(new_w)

        # Nearest neighbor for input_image since we can't interpolate across discontinuities in uv coordinates
        input_image = transform.resize(input_image, (new_h, new_w), order=0)
        color_image = transform.resize(color_image, (new_h, new_w), order=1, anti_aliasing=(new_h < h))

        return {'uv': input_image, 'color': color_image}


class BorderCrop(object):
    def __init__(self, crop_pixels_lr, crop_pixels_tb):
        self.crop_pixels_lr = crop_pixels_lr
        self.crop_pixels_tb = crop_pixels_tb

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']

        # Assuming input_image and color_image are the same shape
        h, w, _ = color_image.shape

        # Get a valid starting and end positions
        h_start = self.crop_pixels_tb
        w_start = self.crop_pixels_lr
        h_end = h - self.crop_pixels_tb
        w_end = w - self.crop_pixels_lr

        # Crop the input and target
        input_image = input_image[h_start:h_end, w_start:w_end, :]
        color_image = color_image[h_start:h_end, w_start:w_end, :]

        return {'uv': input_image, 'color': color_image}


class RandomCrop(object):
    def __init__(self, crop_size):
        assert isinstance(crop_size, tuple)
        self.crop_size = crop_size

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']
        
        # Assuming input_image and color_image are the same shape
        h, w, c = color_image.shape

        size_crop_h, size_crop_w = self.crop_size

        # Get a valid starting and end positions
        h_start = np.random.randint(0, h - size_crop_h) if size_crop_h < h else 0
        w_start = np.random.randint(0, w - size_crop_w) if size_crop_w < w else 0
        h_end = h_start + size_crop_h
        w_end = w_start + size_crop_w
        
        # Crop the input and target
        input_image = input_image[h_start:h_end, w_start:w_end, :]
        color_image = color_image[h_start:h_end, w_start:w_end, :]

        return {'uv': input_image, 'color': color_image}


class RandomFlip(object):
    def __init__(self, flip_axis):
        self.flip_axis = flip_axis

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']
       
        if np.random.choice(a=[False, True]):
            input_image = np.flip(input_image, axis=self.flip_axis).copy()
            color_image = np.flip(color_image, axis=self.flip_axis).copy()

        return {'uv': input_image, 'color': color_image}


class Normalize(object):
    """Normalize color images between [-1,1]."""

    def __call__(self, sample):
        input_image, color_image = sample['uv'], sample['color']
        # NOTE: Don't normalize input_image. It's just a matrix of coordinates

        color_image = img_as_float32(color_image)
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
                 net_input_height, net_input_width, min_scale_size=None, max_scale_size=None,
                 num_ignore_border_pixels_lr=0, num_ignore_border_pixels_tb=0, slice_start=None, slice_end=None,
                 slice_step=None, num_in_train_step=12, num_in_val_step=3, compressed_input=False, num_workers=1, training=True):

        self.data_dir = data_dir
        self.skip = skip
        self.size = (net_input_height, net_input_width)
        self.num_ignore_border_pixels_lr = num_ignore_border_pixels_lr
        self.num_ignore_border_pixels_tb = num_ignore_border_pixels_tb

        # Default min and max scaling size to the smaller image side
        smaller_side_size = min(net_input_height, net_input_width)
        self.min_scale_size = min_scale_size if min_scale_size is not None else smaller_side_size
        self.max_scale_size = max_scale_size if max_scale_size is not None else smaller_side_size

        assert self.min_scale_size >= smaller_side_size,\
            'min scale size cannot be smaller than the smaller network input size'

        self.compressed_input = compressed_input

        with open(os.path.join(data_dir, data_select_file)) as csv_file:
            data = pd.read_csv(csv_file, delimiter=' ', index_col=None, header=None)
            self.use_indices = np.array(data.values).squeeze()

        self.input_color_filenames = self.load_input_color_filenames(data_dir, uv_folder_name, color_folder_name)
        self.input_color_filenames = [self.input_color_filenames[i] for i in self.use_indices if
                                      slice_start <= i < slice_end]
        self.input_color_filenames = self.input_color_filenames[slice(slice_start, slice_end, slice_step)]

        train_id = 0
        val_id = 1
        self.input_color_filenames = np.array(self.input_color_filenames)
        input_color_indices = [train_id if (i % (num_in_val_step + num_in_train_step)) < num_in_train_step else val_id for i in range(len(self.input_color_filenames))]
        input_color_indices = np.array(input_color_indices)
        self.train_filenames = self.input_color_filenames[input_color_indices == train_id]
        self.val_filenames = self.input_color_filenames[input_color_indices == val_id]
       
        print('Train', len(self.train_filenames))
        print('Val', len(self.val_filenames))

        #train_filenames = self.generate_temporal_train_split(self.input_color_filenames, self.skip)

        # Build train transformation
        train_transforms = [
            BorderCrop(self.num_ignore_border_pixels_lr, self.num_ignore_border_pixels_tb),
            Rescale(self.min_scale_size, self.max_scale_size),
            RandomCrop(self.size),
            #RandomFlip(flip_axis=1),
            Normalize(),
            ToTensor()
        ]

        self.dataset = UVDataset(self.train_filenames, compressed_input=self.compressed_input,
                                 transform=transforms.Compose(train_transforms))

        super().__init__(self.dataset, batch_size, shuffle, num_workers, pin_memory=True)
        pass

    def split_validation(self):
        self.val_filenames = self.generate_temporal_val_split(self.input_color_filenames, self.skip)

        # TODO: NOTE: Validation is scaled so that it can fit into memory with the same batch size as the training
        #  dataset. This is a problem because 1) the validation set is not capable of evaluating improvements in
        #  fine details of the neural textures since they're downsampled away and 2) the validation loss between
        #  models trained with different self.size will not be directly comparable. A solution for (1) is to employ
        #  random cropping (but not scaling) on the validation set and a solution for (2) is to use a small enough
        #  batch size for validation to fit the unscaled data into memory.
        # Build val transformation
        val_transforms = [
            BorderCrop(self.num_ignore_border_pixels_lr, self.num_ignore_border_pixels_tb),
            Rescale(self.max_scale_size, self.max_scale_size),
            RandomCrop(self.size), # Added to help data fit into GPU memory.
            Normalize(),
            ToTensor()
        ]

        val_dataset = UVDataset(self.val_filenames, compressed_input=self.compressed_input,
                                transform=transforms.Compose(val_transforms))

        batch_size = self.init_kwargs['batch_size']
        num_workers = self.init_kwargs['num_workers']
        return DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)

    def load_input_color_filenames(self, data_dir, uv_folder_name, color_folder_name):
        input_filenames = self.load_filenames_sorted(data_dir, uv_folder_name)
        color_filenames = self.load_filenames_sorted(data_dir, color_folder_name)

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
