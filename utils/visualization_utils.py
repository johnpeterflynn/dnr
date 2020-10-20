import torch
import matplotlib.pyplot as plt
import glob
import gzip
import json
import os
import re
import numpy as np
import torch.nn.functional as F
from data_loaders import scannet_render_loader

# torch.set_printoptions(precision=10)

# TODO these visualization functions are in deperate need of refactoring

# Visualize a model prediction
def generate_images(model, test_input, tar, cuda=False):
    if cuda:
        model, test_input, tar = model.cuda(), test_input.cuda(), tar.cuda()

    prediction = model(test_input)  # , training=True)
    plt.figure(figsize=(20, 20))

    _, h, w, c = test_input.shape
    test_input_color = torch.zeros((h, w, 3))  # , dtype=type(test_input))
    test_input_color = test_input[:, :, :, 0].cpu()
    tar = tar.permute(0, 2, 3, 1).cpu()
    prediction = prediction.detach().permute(0, 2, 3, 1).cpu()

    display_list = [test_input_color[0].numpy(), tar[0].numpy(), prediction[0].numpy()]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.show()


def generate_comparison(display_images, title, title_color):
    for i, image in enumerate(display_images):
        display_images[i] = image.permute(0, 2, 3, 1)

    # Should assert that rows * cols == len(title) == len(display_images)
    rows, cols = len(display_images), 1
    plt.figure(figsize=(35 * cols, 30 * rows))
    for i in range(len(display_images)):
        plt.subplot(len(display_images), cols, i + 1)
        plt.title(title[i], color=title_color[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_images[i][0].numpy() * 0.5 + 0.5)
        # plt.axis('off')
    plt.show()


def generate_mae(display_images, target, title):
    diffs = []
    target = target.permute(0, 2, 3, 1)
    for i, image in enumerate(display_images):
        display_images[i] = image.permute(0, 2, 3, 1)
        diff = torch.abs(target - display_images[i])
        diffs.append(diff)

    print(len(display_images))

    # Should assert that rows * cols == len(title) == len(display_images)
    rows, cols = len(display_images), 2
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.title("Ground truth")
    plt.imshow(target[0].numpy() * 0.5 + 0.5)
    plt.figure(figsize=(8 * cols, 8 * rows))
    for i in range(0, len(display_images)):
        plt.subplot(rows, cols, 2 * i + 1)
        plt.title(title[i])
        print(display_images[i][0].shape)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_images[i][0].numpy() * 0.5 + 0.5)

        plt.subplot(rows, cols, 2 * i + 2)
        plt.title('MAE: {:.2f}, RGB MSE: {:.2f}'.format(torch.mean(diffs[i]), torch.mean(diffs[i] ** 2) * (
                    255 / 2) ** 2))  # torch.mean(diffs[i])))
        plt.imshow(1.0 - (diffs[i][0] / 2).numpy(), cmap='gray')
    plt.show()


def visualize_texture(neural_texture, select=None, disp_channels=None):
    mipmap = [neural_texture.mipmap_0,
              neural_texture.mipmap_1,
              neural_texture.mipmap_2,
              neural_texture.mipmap_3]

    _, _, _, size = mipmap[0].shape
    sample = 0
    for i, texture in enumerate(mipmap):
        if select is not None and i != select:
            continue
        sample += F.interpolate(texture.detach(), size=size, mode='bilinear', align_corners=False)

    sample = sample[0, :, :, :]
    sample = sample.permute(1, 2, 0)

    height, width, channels = sample.shape

    if disp_channels is None:
        disp_channels = channels

    plt.figure(figsize=(20, 20))
    for i in range(disp_channels):
        plt.title('Channel {}'.format(i))
        plt.subplot(np.ceil(channels / 4), 4, i + 1)
        sample_np = sample[:, :, i].numpy()
        print('min:', np.min(sample_np), 'max:', np.max(sample_np), 'mean:', np.mean(sample_np), 'std:',
              np.std(sample_np))
        plt.imshow(sample_np * 0.5 + 0.5)
        # plt.imshow(sample[:, :, i].numpy() * 0.5 + 0.5)
    plt.show()


def visualize_color_texture(neural_texture):
    mipmap = [neural_texture.mipmap_0,
              neural_texture.mipmap_1,
              neural_texture.mipmap_2,
              neural_texture.mipmap_3]

    _, _, _, size = mipmap[0].shape
    sample = 0
    for i, texture in enumerate(mipmap):
        sample += F.interpolate(texture[:, 0:3, :, :].detach(), size=size, mode='bilinear', align_corners=False)

    sample = sample[0, :, :, :]
    sample = sample.permute(1, 2, 0)

    rescale = 2 / (torch.min(sample) - torch.max(sample))
    sample = sample * rescale
    print(torch.min(sample), torch.max(sample))

    plt.figure(figsize=(20, 20))
    plt.title('Color')
    # plt.subplot(1, 1, 1)
    plt.imshow(sample.numpy() * 0.5 + 0.5)
    plt.show()


## Visualize textures for each given model ##
def visualize_all_textures(models, train_ids):
    num_layers = 4
    for train_id in train_ids:
        print('==================== \/ == Model', train_id, '== \/ ====================')
        visualize_color_texture(models[train_id].neural_texture)
        for layer in range(num_layers):
            print('--> Layer:', layer)
            visualize_texture(models[train_id].neural_texture, select=layer, disp_channels=4)


def display_benchmarks(jit_models, benchmarks):
    # Consts that should be defined globally or inferred
    _UV_CHANNELS = 2
    _SCR_HEIGHT = 968
    _SCR_WIDTH = 1296

    for b_num, uv_file in enumerate(benchmarks):
        # Load UV map
        with gzip.open(uv_file, 'rb') as f:
            uv_image = np.frombuffer(f.read(), dtype='float32')
        uv_image = np.reshape(uv_image, (_SCR_HEIGHT, _SCR_WIDTH, _UV_CHANNELS))
        uv_image = np.flip(uv_image, axis=0).copy()

        print('==Test on UV Map {}/{}=='.format(b_num + 1, len(benchmarks)))

        # Display UV map
        plt.figure(figsize=(8, 8))
        plt.title("Ground truth")
        h, w, c = uv_image.shape
        uv_image_color = np.zeros((h, w, 3))  # , dtype=type(test_input))
        uv_image_color[:, :, 0:2] = uv_image
        plt.imshow(uv_image_color * 0.5 + 0.5)

        # Format for inference in PyTorch
        uv_image = torch.from_numpy(uv_image).unsqueeze(0)

        # Compute and display model prediction
        plt.figure(figsize=(8 * len(jit_models), 8))
        for counter, model in enumerate(jit_models):
            print('inference on jit model', counter)
            prediction = model(uv_image).detach()
            prediction = prediction.squeeze().permute(1, 2, 0)
            plt.subplot(1, len(jit_models), 1 + counter)
            plt.title('Model {}'.format(counter))
            plt.imshow((prediction.numpy() + 1.0) / 2.0)
        plt.show()


# TODO: Remove dataloader creation
## Show a single validation input, ground truth and preducted sample from the first model ##
def display_uv_ground_truth_predicted(models, train_ids, scene, uv_folder_name, filter_file):
    loader = UVDataLoader('data/' + scene, uv_folder_name, 'color', filter_file, 1, shuffle=False,
                          skip=6, slice_start=0, slice_end=5000, compressed_input=True,
                         net_input_height=968, net_input_width=1296, min_crop_scale=1)
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx % 100 != 0:
            continue
        print(batch_idx)
        for train_id in train_ids:
            print('Train ID:', train_id)
            model = models[train_id]
            generate_images(model, data, target)
        #break


# TODO: Remove dataloader creation
## Display training or validation ground truth, prediction and visualize the RGB MAE between them ##
def display_prediction_mae(models, train_ids, titles, frame_index, scene, uv_folder_name, filter_file, validation):
    display_images = []

    # Load from the validatiom dataset
    loader = UVDataLoader('data/' + scene, uv_folder_name, 'color', filter_file, 1, shuffle=True,
                          slice_start=0, slice_end=500, skip=6, compressed_input=True,
                          net_input_height=968, net_input_width=1296, min_crop_scale=1.0, max_crop_scale=1.0)
    if validation:
        loader = loader.split_validation()

    # count = 0
    for batch_idx, (data, target) in enumerate(loader):
        for i, train_id in enumerate(train_ids):
            # if batch_idx < frame_index:
            #    continue

            # Get the trained model
            model = models[train_id]

            # Make a prediction using the model
            prediction = model(data)
            prediction = prediction.detach()
            display_images.append(prediction)
        break
        # print(count)
        # if count < 5:
        #    count += 1
        # else:
        #    count = 0
        #    break

    # Plot results
    generate_mae(display_images, target.detach(), titles)

# TODO: Remove dataloader creation
# TODO: Check, does this still even work?
# Show a validation sample prediction for each model #
def display_prediction_mse(models, traid_ids, scene, uv_folder_name, minc, maxc, height, width, filter_file):
    # display_images = []

    # Load from the validatiom dataset
    loader = UVDataLoader('data/' + scene, uv_folder_name, 'color', filter_file, 1, num_workers=8, shuffle=False,
                          slice_start=0, slice_end=5578, slice_step=5, skip=6, compressed_input=True,
                          net_input_height=256, net_input_width=342, min_crop_scale=minc,
                          max_crop_scale=maxc)  # .split_validation()
    # count = 0
    loss = 0
    mmse = 0
    count = 0
    # Get the trained model
    model = models[train_ids[0]].cuda()
    model = model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        # if batch_idx < frame_index:
        #    continue

        # Make a prediction using the model
        prediction = model(data)
        # display_images.append(prediction)
        diff = torch.abs(target - prediction)
        mae = torch.mean(diff)
        mse = torch.mean(diff ** 2) * (255 / 2) ** 2
        mae = mae.item()
        mse = mse.item()
        loss += mae
        mmse += mse
        count += 1
        # print('Batch {}: MAE: {:.2f}, RGB MSE: {:.2f}'.format(batch_idx, mae, mse))

    print('Loss {:.4f}, MMSE: {:.4f}'.format(loss / count, mmse / count))
    # print(count)
    # if count < 5:
    #    count += 1
    # else:
    #    count = 0
    #    break

    model = model.cpu()
    del data
    del target
    del prediction
    del diff
    del loss
    del mmse
    torch.cuda.empty_cache()
    # Plot results
    #generate_mae(display_images, target.detach(), titles)


# TODO: Remove dataloader creation
def display_prediction_target_metrics(models, train_ids, scene, uv_folder_name, filter_file):
    loader = UVDataLoader('data/' + scene, uv_folder_name, 'color', filter_file, 5, num_workers=8, shuffle=False,
                          slice_start=0, slice_end=12, slice_step=5, skip=6, compressed_input=True,
                          net_input_height=968, net_input_width=1296)

    train_id = train_ids[0]
    for batch_idx, (data, target) in enumerate(loader):
        print(batch_idx)
        print('Train ID:', train_id)
        model = models[train_id]
        prediction = model(data).detach()

        metric = psnr(prediction, target)

        # generate_images(model, data, target)

        print('PSNR:', metric)
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 2, 1)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(target[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.subplot(1, 2, 2)
        plt.imshow(prediction[0].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        plt.show()