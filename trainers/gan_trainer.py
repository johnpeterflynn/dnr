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

        self.set_requires_grad(self.model, False) # Assuming using a pretrained model

        # TODO: Select resnet size, num of input channels, whether to use dropout
        self.netG = gan_networks.define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks', norm='instance',
                use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[self.device]).to(self.device)
        # TODO: Select num of input channels
        self.netD = gan_networks.define_D(input_nc=3 + 3 + 1, ndf=64, netD='basic', norm='instance', init_gain=0.02,
                                          gpu_ids=[self.device]).to(self.device)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Num Generator Parameters:', count_parameters(self.netG))
        print('Num Discriminator Parameters:', count_parameters(self.netD))

        self.criterionGAN = gan_networks.GANLoss(gan_mode='lsgan').to(self.device)

        # TODO: Select optimal lr
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config['optimizer']['args']['lr'], betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config['optimizer']['args']['lr'], betas=(0.5, 0.999))

        # Lazy loading since we're not creating these nets in BaseTrainer. Order in these lists must bethe same as
        # in _lazy_resume_checkpoint()
        if hasattr(self, 'resume_path'):
            self._lazy_resume_checkpoint([self.netG, self.netD], [self.optimizer_G, self.optimizer_D])

        self.train_metrics = MetricTracker('loss_D_fake', 'loss_D_real', 'loss_G', 'loss_D', 'loss_G_fake', 'loss_G_influencer', 'acc_D_real', 'acc_D_fake', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
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
        #self.optimizer.zero_grad()
        self.optimizer_G.zero_grad()
        self._backward_G()
        #self.optimizer.step()
        self.optimizer_G.step()

    def _forward(self):
        # TODO: NOTE: self.real_color should be generated elsewhere, either from a dataloader
        #  or from a different module that manages self.model
        self.real_color = self.model(self.real_uv).detach()
        self.prior_color = self._create_masked_image(self.real_color, self.mask)
        self.fake_color = self.netG(self.prior_color)

    def _create_masked_image_square(self, image, h_norm_start, w_norm_start, h_norm_end, w_norm_end):
        _, _, h, w = image.shape
        h_start, w_start, h_end, w_end = int(h_norm_start * h), int(w_norm_start * w), int(h_norm_end * h), int(w_norm_end * w)
        masked_image = image.clone()
        masked_image[:, :, h_start:h_end, w_start:w_end] = 1.0  # White

        return masked_image

    def _create_masked_image(self, image, mask):
        masked_image = image.clone()
        masked_image[mask.expand_as(masked_image)] = 1.0  # Set to white (TODO: how else to mask?)

        return masked_image

    def _backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_input = torch.cat((self.mask, self.prior_color, self.fake_color), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_input.detach())
        self.accuracy_D_fake = torch.mean(1 - torch.sigmoid(pred_fake)).item()
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_input = torch.cat((self.mask, self.prior_color, self.real_color), 1)
        pred_real = self.netD(real_input)
        self.accuracy_D_real = torch.mean(torch.sigmoid(pred_real)).item()
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def _backward_G(self):
        """Calculate GAN and other losses for the generator"""
        # First, G(A) should fake the discriminator
        # Swap uv dimensions since they are initially in a different order for grid_sample()
        fake_input = torch.cat((self.mask, self.prior_color, self.fake_color), 1)
        pred_fake = self.netD(fake_input)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        # TODO: Find good value for self.lambda_other_criterion
        self.lambda_other_criterion = 10.0
        self.loss_G_influencer = self.criterion(self.fake_color, self.real_color) * self.lambda_other_criterion
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_influencer
        self.loss_G.backward()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.eval()
        self.train_metrics.reset()
        for batch_idx, (real_uv_cpu, _, mask_cpu) in enumerate(self.data_loader):
            self.real_uv = real_uv_cpu.to(self.device)
            self.mask = mask_cpu.to(self.device)

            # Assigned in forward for now since we're doing masking there
            #self.real_color = real_color_cpu.to(self.device, non_blocking=True)

            self._optimize_parameters()

            self.train_metrics.update('loss_G', self.loss_G.item(), write=False)
            self.train_metrics.update('loss_D', self.loss_D.item(), write=False)
            self.train_metrics.update('loss_D_fake', self.loss_D_fake.item(), write=False)
            self.train_metrics.update('loss_D_real', self.loss_D_real.item(), write=False)
            self.train_metrics.update('loss_G_fake', self.loss_G_GAN.item(), write=False)
            self.train_metrics.update('loss_G_influencer', self.loss_G_influencer.item(), write=False)
            self.train_metrics.update('acc_D_real', self.accuracy_D_real, write=False)
            self.train_metrics.update('acc_D_fake', self.accuracy_D_fake, write=False)
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(self.fake_color, self.real_color), write=False)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} G Loss: {:.6f} D Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    self.loss_G.item(), self.loss_D.item()))

            if batch_idx == self.len_epoch:
                break

        self.writer.set_step(epoch - 1)
        log = self.train_metrics.result(write=True)

        # Only visualize the final sample for brevity
        self._visualize_input(real_uv_cpu)
        self._visualize_prior(self.prior_color.cpu())
        self._visualize_prediction(self.fake_color.cpu())
        self._visualize_target(self.real_color.cpu())

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
            for batch_idx, (real_uv_cpu, _, mask_cpu) in enumerate(self.valid_data_loader):
                self.real_uv = real_uv_cpu.to(self.device)
                self.mask = mask_cpu.to(self.device)

                # Assigned in forward for now since we're doing masking there
                #self.real_color = real_color_cpu.to(self.device, non_blocking=True)

                self._forward()

                #output = self.model(data)
                # TODO: Remove explicit specification of loss functions
                #loss = self.criterionVGG(output, target) + self.criterion(output, target)
                loss = self.criterion(self.fake_color, self.real_color)

                self.valid_metrics.update('loss', loss.item(), write=False)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(self.fake_color, self.real_color), write=False)

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        self.writer.set_step(epoch - 1, 'valid')
        log = self.valid_metrics.result(write=True)

        # Only visualize the final sample for brevity
        self._visualize_input(real_uv_cpu)
        self._visualize_prior(self.prior_color.cpu())
        self._visualize_prediction(self.fake_color.cpu())
        self._visualize_target(self.real_color.cpu())

        return log

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch_G = type(self.netG).__name__
        arch_D = type(self.netD).__name__
        state = {
            'archs': [arch_G, arch_D],
            'epoch': epoch,
            'state_dicts': [self.netG.state_dict(), self.netD.state_dict()],
            'optimizer_state_dicts': [self.optimizer_G.state_dict(), self.optimizer_D.state_dict()],
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best(self, epoch):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch_G = type(self.netG).__name__
        arch_D = type(self.netD).__name__
        state = {
            'archs': [arch_G, arch_D],
            'epoch': epoch,
            'state_dicts': [self.netG.state_dict(), self.netD.state_dict()],
            'optimizer_state_dicts': [self.optimizer_G.state_dict(), self.optimizer_D.state_dict()],
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.resume_path = str(resume_path)

    # TODO: PROBLEM: base class tries to resume checkpoint before derived class creates models
    def _lazy_resume_checkpoint(self, models, optimizers):
        self.logger.info("Loading checkpoint: {} ...".format(self.resume_path))
        checkpoint = torch.load(self.resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        for index, net in enumerate(models):
            # load architecture params from checkpoint.
            net.load_state_dict(checkpoint['state_dicts'][index])
            self.logger.info("Loaded model {}".format(index))

        for index, opt in enumerate(optimizers):
            # load optimizer state from checkpoint only when optimizer type is not changed.
            opt.load_state_dict(checkpoint['optimizer_state_dicts'][index])
            self.logger.info("Loaded optimizer {}".format(index))

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

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

    def _visualize_prior(self, data):
        """format and display output data on tensorboard"""
        self.writer.add_image('prior', make_grid(data[0,:,:,:].unsqueeze(0), nrow=8, normalize=True))

    def _visualize_prediction(self, data):
        """format and display output data on tensorboard"""
        self.writer.add_image('predicted', make_grid(data[0,:,:,:].unsqueeze(0), nrow=8, normalize=True))

    def _visualize_target(self, target):
        """format and display target data on tensorboard"""
        self.writer.add_image('target', make_grid(target[0,:,:,:].unsqueeze(0), nrow=8, normalize=True))
