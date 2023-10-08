import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from models import networks
from torchvision import models
from collections import namedtuple


def create_model(opt):
    print(opt.model)
    if opt.model == 'AECR_SPP':
        model = AECR()
    else:
        raise ValueError("Model [%s] creation failed." % opt.model)
    model.initialize(opt)
    print("model [%s] was successfully created/loaded" % (model.name()))
    return model


class Feature(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Feature, self).__init__()
        pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(5):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(5, 10):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(10, 17):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(17, 23):
            self.slice4.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class BaseModel:
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class AECR(BaseModel):

    def name(self):
        return 'AECR_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain

        self.Feature_ex = Feature(requires_grad=False)
        self.Feature_ex = self.Feature_ex.cuda()
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(
            self.opt.fineSize / 4) - self.opt.overlap, \
        int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(
            self.opt.fineSize / 4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        self.netG, self.Cosis_list, self.Cosis_list2, self.AECR_model = networks.define_G(opt.input_nc_g, opt.output_nc,
                                                                                          opt.ngf,
                                                                                          opt.which_model_netG, opt,
                                                                                          self.mask_global, opt.norm,
                                                                                          opt.use_dropout,
                                                                                          opt.init_type, self.gpu_ids,
                                                                                          opt.init_gain)
        self.netP, _, _, _ = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                               opt.which_model_netP, opt, self.mask_global, opt.norm, opt.use_dropout,
                                               opt.init_type, self.gpu_ids, opt.init_gain)
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True

            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)
            self.netF = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netF,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netP, 'P', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_P)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('Networks initialized')
            networks.print_network(self.netG)
            networks.print_network(self.netP)
            if self.isTrain:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
            print('-----------------------------')

    def set_input(self, input, mask):
        input_A = input
        input_B = input.clone()
        input_mask = mask
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = 0
        if self.opt.mask_type == 'center':
            self.mask_global = self.mask_global
        elif self.opt.mask_type == 'random':
            self.mask_global.zero_()
            self.mask_global = input_mask
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        self.ex_mask = self.mask_global.expand(1, 3, self.mask_global.size(2), self.mask_global.size(3))

        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).byte()
        self.input_A.narrow(1, 0, 1).masked_fill_(self.mask_global.bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 1, 1).masked_fill_(self.mask_global.bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 2, 1).masked_fill_(self.mask_global.bool(), 2 * 117.0 / 255.0 - 1.0)
        self.set_latent_mask(self.mask_global, 3, self.opt.threshold)

    def set_latent_mask(self, mask_global, layer_to_last, threshold):
        self.AECR_model[0].set_mask(mask_global, layer_to_last, threshold)
        self.Cosis_list[0].set_mask(mask_global, self.opt)
        self.Cosis_list2[0].set_mask(mask_global, self.opt)

    def forward(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)
        self.un = self.fake_P.clone()
        self.Unknowregion = self.un.data.masked_fill_(self.inv_ex_mask.bool(), 0)
        self.knownregion = self.real_A.data.masked_fill_(self.ex_mask.bool(), 0)
        self.Syn = self.Unknowregion + self.knownregion
        self.Middle = torch.cat((self.Syn, self.input_A), 1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)

    def set_gt_latent(self):
        gt_latent = self.Feature_ex(Variable(self.input_B, requires_grad=False))
        self.Cosis_list[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent.relu4_3)

    def test(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)
        self.un = self.fake_P.clone()
        self.Unknowregion = self.un.data.masked_fill_(self.inv_ex_mask.bool(), 0)
        self.knownregion = self.real_A.data.masked_fill_(self.ex_mask.bool(), 0)
        self.Syn = self.Unknowregion + self.knownregion
        self.Middle = torch.cat((self.Syn, self.input_A), 1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)

    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        self.gt_latent_fake = self.Feature_ex(Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real = self.Feature_ex(Variable(self.input_B, requires_grad=False))
        real_AB = self.real_B
        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)
        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)
        self.loss_D = self.loss_D_fake * 0.5 + self.loss_F_fake * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = self.fake_B
        fake_f = self.gt_latent_fake
        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(fake_f.relu3_3)
        pred_real = self.netD(self.real_B)
        pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F,
                                                                                             False)
        self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B) + self.criterionL1(self.fake_P,
                                                                                        self.real_B)) * self.opt.lambda_A
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight
        self.ng_loss_value = 0
        self.ng_loss_value2 = 0
        if self.opt.cosis:
            for gl in self.Cosis_list:
                self.ng_loss_value += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value
            for gl in self.Cosis_list2:
                self.ng_loss_value2 += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value2

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_G.zero_grad()
        self.optimizer_P.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_P.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D', self.loss_D_fake.data.item()),
                            ('F', self.loss_F_fake.data.item())
                            ])

    def get_current_visuals(self):

        real_A = self.real_A.data
        fake_B = self.fake_B.data
        real_B = self.real_B.data

        return real_A, real_B, fake_B

    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.gpu_ids)
        self.save_network(self.netP, 'P', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        self.save_network(self.netF, 'F', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
        self.load_network(self.netP, 'P', epoch)
