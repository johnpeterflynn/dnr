from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import gzip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip
from utils import vector_math


##-- Visualize Texture Coords --##
_SCREEN_HEIGHT, _SCREEN_WIDTH, _UV_CHANNELS = 968, 1296, 2


def get_train_val_split(pose_files, skip, max_index=None, stride=1):
    if max_index is None:
        max_index = len(pose_files)
    pose_files = pose_files[0:max_index:stride]
    train_filenames = [pose_files[i] for i in range(len(pose_files)) if (i % skip) != 0]
    val_filenames = [pose_files[i] for i in range(len(pose_files)) if (i % skip) == 0]

    return train_filenames, val_filenames


def load_poses(pose_files):
    rots = []
    ts = []
    for file in pose_files:
        with open(file) as csv_file:
            data = pd.read_csv(csv_file, delimiter=' ', index_col=None, header=None)

            rot = np.array(data.values[0:3, 0:3])
            t = np.array(data.values[0:3, -1])
            rots.append(rot)
            ts.append(t)
    return rots, ts


def get_nn_indices(neighbors, items):
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(neighbors)

    neigh_dist, neigh_index = neigh.kneighbors(items)

    return neigh_index


def get_rotvecs_from_matrices(rots):
    rotvecs = []
    for rot in rots:
        rotvecs.append(R.from_matrix(rot).as_rotvec())

    return rotvecs


# Load train and validation poses
def get_val_nn_train_angles(train_rots, val_rots, unit='deg'):
    # Transform to axis representation
    train_rotvecs = get_rotvecs_from_matrices(train_rots)
    val_rotvecs = get_rotvecs_from_matrices(val_rots)

    # Find nearest neighbors by angle
    nn_indices = get_nn_indices(train_rotvecs, val_rotvecs)

    # Get angles to nearest neighbords in degrees
    angles = []
    for i, val_rotvec in enumerate(val_rotvecs):
        angle = vector_math.angle_between(val_rotvec, train_rotvecs[nn_indices[i, 0]])
        if unit == 'deg':
            angle = np.rad2deg(angle)
        angles.append(angle)

    return angles


def load_texture_coord(file, compressed=True):
    if compressed:
        # Decompress texture coordinate file into a numpy array
        with gzip.open(file, 'rb') as f:
            uv_image = np.frombuffer(f.read(), dtype='float32')
    else:
        uv_image = np.fromfile(uv_image_path, dtype='float32')

    uv_image = np.reshape(uv_image, (_SCREEN_HEIGHT, _SCREEN_WIDTH, _UV_CHANNELS))
    uv_image = np.flip(uv_image, axis=0).copy()

    return uv_image


def visualize_texture_coord(uv_image):
    uv_color_image = np.zeros((_SCREEN_HEIGHT, _SCREEN_WIDTH, _UV_CHANNELS))
    uv_color_image[:, :, 0:_UV_CHANNELS] = uv_image

    # Should assert that rows * cols == len(title) == len(display_images)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(uv_color_image)
    plt.show()


"""
def visualize_texture_coord_mask(uv_image):
    uv_color_image = uv_image[:, :, -1]

    # Should assert that rows * cols == len(title) == len(display_images)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 1, 1)
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(uv_color_image, cmap='binary')
    plt.show()
"""