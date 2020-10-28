import argparse
import collections
import torch
import numpy as np
import data_loaders as module_data
import models.losses as module_loss
import models.metric as module_metric
import models as module_arch
from parse_config import ConfigParser
from trainers import RenderTrainer
import subprocess


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
np.random.seed(SEED)


def main(config):
    # init logger
    logger = config.get_logger('train')

    # log the random seed
    logger.info("Random seed: {}".format(SEED))

    # log the current git hash
    logger.info("Git hash: {}".format(
        subprocess.check_output(["git", "describe", "--always"]).strip()))

    # print training session description to logs
    logger.info("Description: {}".format(config["description"]))

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    trainer = RenderTrainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--dry_run', default=False, type=bool,
                      help='If true, disables logging of models to disk and tags to git (default: False)')
    args.add_argument('-m', '--message', default=None, type=str,
                      help='description of this training session')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--ld', '--log_dir'], type=str, target='trainer;save_dir')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
