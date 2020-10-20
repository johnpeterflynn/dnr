import torch
import matplotlib.pyplot as plt
import glob
import gzip
import json
import os
import re
from models import RenderNet

# Load a trained model using its best weights unless otherwise specified
def load_trained_model(train_id, logs='./saved', checkpoint_name='model_best'):
    model_path = os.path.join(logs, 'models/DNR', train_id)

    # Load config file
    config_file = os.path.join(model_path, "config.json")
    if config_file:
        with open(config_file, 'r') as f:
            config = json.load(f)

    # Load model weights
    checkpoint_path = os.path.join(model_path, checkpoint_name) + '.pth'

    if not os.path.isfile(checkpoint_path):
        pose_files = glob.glob(os.path.join(model_path, '*.pth'))

        def extract_epoch(f):
            s = re.findall("checkpoint-epoch(\d+)", f)
            return (int(s[0]) if s else -1, f)

        checkpoint_path = max(pose_files, key=extract_epoch)

    print('loaded:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    loaded_epoch = checkpoint['epoch'] + 1
    print('Loaded checkpoint from epoch', loaded_epoch)

    # Handling loading of models from different versions will be tricky.
    # Eventually might need to check models out from git repo. For now,
    # use these fixes.
    if 'mipmap_levels' in config['arch']['args']:
        mipmap_levels = config['arch']['args']['mipmap_levels']
        zero_other_mipmaps = False
        print('contains mipmap:', mipmap_levels)
    else:
        mipmap_levels = 1
        checkpoint['state_dict']['neural_texture.mipmap.0'] = checkpoint['state_dict'].pop('neural_texture.texture')
        zero_other_mipmaps = True
        print('Warning: {} weights from old model version with only one mipmap layer'.format(train_id))

    # Ahh! We have to hack the weights since ParameterList isn't accepted by TorchScript
    if 'neural_texture.mipmap.0' in checkpoint['state_dict']:
        print('Warning: {} mipmap from old model version that used ParameterList or a single layer'.format(train_id))
        for i in range(mipmap_levels):
            checkpoint['state_dict']['neural_texture.mipmap_{}'.format(i)] = checkpoint['state_dict'].pop(
                'neural_texture.mipmap.{}'.format(i))

    # Load model with parameters from config file
    model = RenderNet(config['arch']['args']['texture_size'],
                      config['arch']['args']['texture_depth'],
                      mipmap_levels)

    # TODO: WARNING: Leaving some mipmap layer weights unassigned might lead to erroneous
    #  results (maybe they're not set to zero by default)
    # Assign model weights and set to eval (not train) mode
    model.load_state_dict(checkpoint['state_dict'], strict=(not zero_other_mipmaps))
    model.eval()

    return model, loaded_epoch


# load all models by train id
def load_trained_models(train_ids, logs='./saved'):
    models = {}
    for train_id in train_ids:
        models[train_id], _ = load_trained_model(train_id, logs=logs)

    return models


# Cereate a libtorch script file containing the model that can be loaded into C++
def create_libtorch_script(train_id, logs, checkpoint_name='model_best', model_script_folder='./libtorch-models'):
    model, loaded_epoch = load_trained_model(train_id, logs=logs, checkpoint_name=checkpoint_name)
    sm = torch.jit.script(model)
    model_script_name = 'DNR-{}-{}-epoch-{}_model.pt'.format(train_id, checkpoint_name, loaded_epoch)
    model_script_path = os.path.join(model_script_folder, model_script_name)
    print(model_script_path)
    sm.save(model_script_path)


# Load a libtorch script file
def load_libtorch_script(train_id=None, model_script_name=None, checkpoint_name='model_best',
                         model_script_folder='./libtorch-models'):
    assert train_id is not None or model_script_name is not None

    if model_script_name is None:
        model_script_name = 'DNR-{}-{}_model.pt'.format(train_id, checkpoint_name)

    model_script_path = os.path.join(model_script_folder, model_script_name)

    print(model_script_path)
    sm_loaded = torch.jit.load(model_script_path)
    sm_loaded.eval()
    return sm_loaded


def get_sorted_files(files):
    """Given a Unix-style pathname pattern, returns a sorted list of files matching that pattern"""
    files = glob.glob(files)
    files.sort()
    return files

