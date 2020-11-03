import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from models.losses import VGGLoss, laplacian_pyramid_l2_regularization
from utils import inf_loop, MetricTracker
from models import gan_networks


class GANTrainer(BaseTrainer):
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

        self.netD = gan_networks.define_D(input_nc=2 + 3, ndf=64, netD='basic', norm='instance', init_gain=0.02,
                                          gpu_ids=[self.device]).to(self.device)
        self.criterionGAN = gan_networks.GANLoss(gan_mode='lsgan').to(self.device)

        #self.optimizer_G = torch.optim.Adam(self.model.parameters(), lr=config['optimizer']['args']['lr'], betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config['optimizer']['args']['lr'], betas=(0.5, 0.999))

        self.train_metrics = MetricTracker('loss_G', 'loss_D', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    # TODO: WARNING: CAN this interfere with loss functions with weights?
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _optimize_parameters(self):
        self._forward()

        # TODO: Q: Shouldn't we turn off the generator's gradients here?
        # Discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self._backward_D()
        self.optimizer_D.step()

        # Generator
        self.set_requires_grad(self.netD, False)
        self.optimizer.zero_grad()
        self._backward_G()
        self.optimizer.step()

    def _forward(self):
        self.fake_color = self.model(self.real_uv)

    def _backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        real_uv_permuted = self.real_uv.permute(0, 3, 1, 2)
        fake_uv_color = torch.cat((real_uv_permuted, self.fake_color), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_uv_color.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_uv_color = torch.cat((real_uv_permuted, self.real_color), 1)
        pred_real = self.netD(real_uv_color)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def _backward_G(self):
        """Calculate GAN and other losses for the generator"""
        # First, G(A) should fake the discriminator
        # Swap uv dimensions since they are intiailly in a different order for grid_sample()
        fake_uv_color = torch.cat((self.real_uv.permute(0, 3, 1, 2), self.fake_color), 1)
        pred_fake = self.netD(fake_uv_color)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # TODO: Find good value for lambda_L1
        self.loss_G_other = self.criterion(self.fake_color, self.real_color) #* 10.0#self.opt.lambda_L1
        print('Generator Loss Gan, other:', self.loss_G_GAN.item(), self.loss_G_other.item())
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_other
        self.loss_G.backward()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (real_uv_cpu, real_color_cpu) in enumerate(self.data_loader):
            self.real_uv = real_uv_cpu.to(self.device)
            self.real_color = real_color_cpu.to(self.device, non_blocking=True)

            self._optimize_parameters()

            print('Loss G', self.loss_G.item())
            print('Loss D', self.loss_D.item())

            self.train_metrics.update('loss_G', self.loss_G.item(), write=False)
            self.train_metrics.update('loss_D', self.loss_D.item(), write=False)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(self.fake_color, self.real_color), write=False)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G Loss: {:.6f} D Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.loss_G.item(), self.loss_D.item()))

            if batch_idx == self.len_epoch:
                break

        # Only visualize the final sample for brevity
        self._visualize_input(real_uv_cpu)
        self._visualize_prediction(self.fake_color.cpu())
        self._visualize_target(real_color_cpu)

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
            for batch_idx, (real_uv_cpu, real_color_cpu) in enumerate(self.valid_data_loader):
                self.real_uv = real_uv_cpu.to(self.device)
                self.real_color = real_color_cpu.to(self.device, non_blocking=True)

                self._forward()

                #output = self.model(data)
                # TODO: Remove explicit specification of loss functions
                #loss = self.criterionVGG(output, target) + self.criterion(output, target)
                loss = self.criterion(self.fake_color, self.real_color)

                self.valid_metrics.update('loss', loss.item(), write=False)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self.fake_color, self.real_color), write=False)

            # Only visualize the final sample for brevity
            self._visualize_input(real_uv_cpu)
            self._visualize_prediction(self.fake_color.cpu())
            self._visualize_target(real_color_cpu)

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
