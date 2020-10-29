import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from models.losses import VGGLoss, laplacian_pyramid_l2_regularization
from utils import inf_loop, MetricTracker


class RenderTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Init model for VGG loss
        #_, device_ids = self._prepare_device(self.config['n_gpu'])
        #self.criterionVGG = VGGLoss().to(self.device, non_blocking=True);
        #self.criterionVGG = torch.nn.DataParallel(self.criterionVGG, device_ids)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data_cpu, target_cpu) in enumerate(self.data_loader):
            data = data_cpu.to(self.device)
            target = target_cpu.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            
            # TODO: Remove explicit specification of loss and regularization functions
            #loss = self.criterionVGG(output, target) + self.criterion(output, target)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item(), write=False)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), write=False)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        # Only visualize the final sample for brevity
        self._visualize_input(data_cpu)
        self._visualize_prediction(output.cpu())
        self._visualize_target(target_cpu)

        self.writer.set_step(epoch - 1)
        log = self.train_metrics.result(write=True)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data_cpu, target_cpu) in enumerate(self.valid_data_loader):
                data = data_cpu.to(self.device)
                target = target_cpu.to(self.device, non_blocking=True)

                output = self.model(data)
                # TODO: Remove explicit specification of loss functions
                #loss = self.criterionVGG(output, target) + self.criterion(output, target)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item(), write=False)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), write=False)

            # Only visualize the final sample for brevity
            self._visualize_input(data_cpu)
            self._visualize_prediction(output.cpu())
            self._visualize_target(target_cpu)

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        self.writer.set_step(epoch - 1, 'valid')
        log = self.valid_metrics.result(write=True)

        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _visualize_input(self, data):
        """format and display input data on tensorboard"""
        # Add am empty dimension to the uv coordinates so that it can be displayed
        # in RGB
        # NOTE: uv coords are in form N x H x W x 2. We need to convert it to image
        # format, eg: N x 3 x H x W
        b, h, w, c = data.shape
        data3 = torch.zeros(b, h, w, c + 1)
        data3[:, :, :, 0:c] = data
        data3 = data3.permute(0, 3, 1, 2)
        self.writer.add_image('input', make_grid(data3[0,:,:,:].unsqueeze(0), nrow=8, normalize=False))

    def _visualize_prediction(self, output):
        """format and display output data on tensorboard"""
        self.writer.add_image('output', make_grid(output[0,:,:,:].unsqueeze(0), nrow=8, normalize=True))

    def _visualize_target(self, target):
        """format and display target data on tensorboard"""
        self.writer.add_image('target', make_grid(target[0,:,:,:].unsqueeze(0), nrow=8, normalize=True))
