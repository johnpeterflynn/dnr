"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
import gzip
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count

import numpy as np
import torch
import torchvision.transforms as TF
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))


class UVPathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i, height, width):
        path = self.files[i]
        _UV_CHANNELS = 2
        with gzip.open(path, 'rb') as f:
            uv_image = np.frombuffer(f.read(), dtype='float32')
        uv_image = np.reshape(uv_image, (height, width, _UV_CHANNELS))
        uv_image = np.flip(uv_image, axis=0).copy()
        if self.transforms is not None:
            uv_image = self.transforms(uv_image)
        uv_image = torch.from_numpy(uv_image)
        return uv_image


class ImagesPathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class FIDScore:
    def __init__(self, dims=2048, device='cpu'):
        """Init FIDScore

        Params:
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        """
        self.dims = dims
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(self.device)
        self.model.eval()


    def get_activations(self, dataset, inf_model=None, batch_size=50):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
        -- dataset     : Dataset containing images or inputs to inf_model
        -- inf_model   : Model in which to input the dataset. Use dataset directly
                         if inf_model is None.
        -- batch_size  : Batch size of images for the model to process at once.
                         Make sure that the number of samples is a multiple of
                         the batch size, otherwise some samples are ignored. This
                         behavior is retained to match the original FID score
                         implementation.

        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """

        if batch_size > len(dataset):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = len(dataset)

        dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         drop_last=False, num_workers=cpu_count())

        pred_arr = np.empty((len(dataset), self.dims))

        start_idx = 0

        for batch in tqdm(dl):
            batch = batch.to(self.device)

            # Use output of inf_model instead of batch if an inf_model is given.
            if inf_model is not None:
                batch = inf_model(batch)
            
            with torch.no_grad():
                pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        return pred_arr


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


    def calculate_activation_statistics(self, dataset, inf_model=None, batch_size=50):
        """Calculation of the statistics used by the FID.
        Params:
        -- dataset     : Dataset containing images or inputs to inf_model
        -- inf_model   : Model in which to input the dataset. Use dataset directly
                         if inf_model is None.
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.

        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(dataset, inf_model, batch_size)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


    def compute_statistics_of_path(self, path, batch_size):
        if path.endswith('.npz'):
            f = np.load(path)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            path = pathlib.Path(path)
            files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
            ds = ImagesPathDataset(files, transforms=TF.ToTensor())
            m, s = self.calculate_activation_statistics(ds, batch_size=batch_size)

        return m, s


    def compute_statistics_of_model(self, inf_path, inf_model, inf_size, batch_size):
        inf_height, inf_width = inf_size
        inf_path = pathlib.Path(inf_path)
        # TODO: Generalize path extension
        files = list(inf_path.glob('*.gz'))
        ds = UVPathDataset(files, inf_height, inf_width)
        m, s = self.calculate_activation_statistics(ds, inf_model, batch_size=batch_size)

        return m, s


    def calculate_fid_given_paths(self, paths, batch_size):
        """Calculates the FID of two paths"""
        for p in paths:
            if not os.path.exists(p):
                raise RuntimeError('Invalid path: %s' % p)

        m1, s1 = self._compute_statistics_of_path(paths[0], batch_size)
        m2, s2 = self._compute_statistics_of_path(paths[1], batch_size)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value


    def calculate_fid_given_path_and_model(self, truth_path, inf_path, inf_model,
            inf_size, batch_size):
        """Calculates the FID given a path to files on disk and a model to
        infer images from.
        """
        for p in [truth_path, inf_path]:
            if not os.path.exists(p):
                raise RuntimeError('Invalid path: %s' % p)

        m1, s1 = self.compute_statistics_of_path(truth_path, batch_size)
        m2, s2 = self.compute_statistics_of_model(inf_path, inf_model, inf_size, batch_size)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value



def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    fid = FIDScore(args.dims, device)

    fid_value = fid.calculate_fid_given_paths(args.path,
                                          args.batch_size)
    print('FID: ', fid_value)


#if __name__ == '__main__':
#    main()
