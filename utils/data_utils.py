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
_SUPPORTED_UV_CHANNELS = 2
_NUM_BITS_PER_UV_COORD = 16


def get_train_val_test_split(data_ids, num_in_train_step, num_in_val_step, data_select_path=None, slice_start=0,
                             slice_step=1, slice_end=None):
    """Separates data_ids into three separate lists of ids for train, val and test respectively.
    """
    train_flag = 0
    val_flag = 1
    test_flag = 2

    if slice_end is None:
        slice_end = len(data_ids)

    if data_select_path is not None:
        with open(data_select_path) as csv_file:
            data = pd.read_csv(csv_file, delimiter=' ', index_col=None, header=None)
            use_indices = np.array(data.values).squeeze()
    else:
        use_indices = np.arange(slice_end)

    data_ids = [data_ids[i] for i in use_indices if slice_start <= i < slice_end]
    data_ids = data_ids[slice(slice_start, slice_end, slice_step)]

    data_ids = np.array(data_ids)
    category_indices = [(test_flag if (i // (num_in_train_step + num_in_val_step)) % 2 == 0 else val_flag)
                        if (i % (num_in_train_step + num_in_val_step)) >= num_in_train_step else train_flag for i
                        in range(len(data_ids))]

    category_indices = np.array(category_indices)
    train_ids = data_ids[category_indices == train_flag]
    val_ids = data_ids[category_indices == val_flag]
    test_ids = data_ids[category_indices == test_flag]

    print('Train', len(train_ids))
    print('Val', len(val_ids))
    print('Test', len(test_ids))

    return train_ids, val_ids, test_ids


# NOTE: DEPRECATED
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


def _uv_significand_to_float(uv_image, channels):
    # Convert significand form back to floating point representation (excluding mask layer)
    uv_image = uv_image.astype(np.float32)
    uv_image[:, :, 0:channels-1] = uv_image[:, :, 0:channels-1] / (2 ** _NUM_BITS_PER_UV_COORD)

    return uv_image


def load_texture_coord(file, height=_SCREEN_HEIGHT, width=_SCREEN_WIDTH, channels=None, compressed=True, encoded=True):
    if encoded:
        dtype = np.ushort
    else:
        dtype = np.float32

    if compressed:
        # Decompress texture coordinate file into a numpy array
        with gzip.open(file, 'rb') as f:
            uv_image = np.frombuffer(f.read(), dtype=dtype)
    else:
        uv_image = np.fromfile(file, dtype=dtype)

    if channels is None:
        channels = int(len(uv_image) / (height * width))

    uv_image = np.reshape(uv_image, (height, width, channels))
    ## TODO: Remove need to flip image by optimizing data preprocessing
    uv_image = np.flip(uv_image, axis=0)

    # If dtype is an unsigned short, assume it is in significand form
    if encoded:
        uv_image = _uv_significand_to_float(uv_image, channels)
    else:
        ## Stride becomes negative without a copy. uv_image.astype() executes a copy.
        uv_image = uv_image.copy()

    if channels > _SUPPORTED_UV_CHANNELS:
        # print("{} channels in UV files but only {} chanels supported. Clipping channels.".format(
        #        num_channels, _SUPPORTED_UV_CHANNELS))
        uv_image = uv_image[:, :, 0:_SUPPORTED_UV_CHANNELS]

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
