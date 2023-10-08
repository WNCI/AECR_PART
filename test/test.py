import time
from models.utils import Data_load
from models.models import create_model
import os
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_srcdata', type=str, default='..//test\Image_Src',
                    help='Path to the sky polarization pattern test file')
parser.add_argument('--test_maskdata', type=str, default='..//test\Image_Mask',
                    help='Path to the corresponding mask for the sky polarization mode test file')
parser.add_argument('--checkpoints_dir', type=str, default=r'..\checkpoints', help="here models are saved "
                                                                                  "during training")
parser.add_argument('--save_dir', type=str, default='../rec', help='Test Result Save Path')
parser.add_argument('--load_epoch', type=int, default=120, help='Loading pre-trained models and parameters')
parser.add_argument('--batchSize', type=int, default=1, help="batch size used during training")
parser.add_argument('--fineSize', type=int, default=256, help="image size")
parser.add_argument('--input_nc', type=int, default=3, help="input channel size for first stage")
parser.add_argument('--input_nc_g', type=int, default=6, help="input channel size for second stage")
parser.add_argument('--output_nc', type=int, default=3, help="output channel size")
parser.add_argument('--ngf', type=int, default=64, help="inner channel")
parser.add_argument('--ndf', type=int, default=64, help="inner channel")
parser.add_argument('--which_model_netD', type=str, default='basic', help="patch discriminator")
parser.add_argument('--which_model_netF', type=str, default='feature', help="feature patch discriminator")
parser.add_argument('--which_model_netG', type=str, default='unet_AECR', help="seconde stage network")
parser.add_argument('--which_model_netP', type=str, default='unet_256', help="first stage network")
parser.add_argument('--triple_weight', type=int, default=1, help="weight")
parser.add_argument('--name', type=str, default='AECR_SPP', help="Model name")
parser.add_argument('--model', type=str, default='AECR_SPP', help="Model name")
parser.add_argument('--n_layers_D', type=str, default='3', help="network depth")
parser.add_argument('--gpu_ids', type=list, default=[0], help='GPU to be used')
parser.add_argument('--norm', type=str, default='instance', help="norm")
parser.add_argument('--fixed_mask', type=int, default=1, help="fixed mask")
parser.add_argument('--use_dropout', type=bool, default=False, help="bool var")
parser.add_argument('--init_type', type=str, default='normal', help="init_type")
parser.add_argument('--mask_type', type=str, default='random', help="mask_type")
parser.add_argument('--shift_sz', type=int, default=1, help="size of feature patch")
parser.add_argument('--threshold', type=float, default=5 / 16.0)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--mask_thred', type=int, default=1)
parser.add_argument('--strength', type=int, default=1)
parser.add_argument('--init_gain', type=float, default=0.02)
parser.add_argument('--gan_type', type=str, default='lsgan')
parser.add_argument('--overlap', type=int, default=4)
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--epoch_count', type=int, default=1)
parser.add_argument('--niter', type=int, default=20)
parser.add_argument('--niter_decay', type=int, default=100)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_policy', type=str, default='lambda')
parser.add_argument('--isTrain', type=bool, default=True)

opt = parser.parse_args()
transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(), ])
transform = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
dataset_test = Data_load(opt.test_srcdata, opt.test_maskdata, transform, transform_mask)
iterator_test = (data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True))
print(len(dataset_test))

model = create_model(opt)
model.load(opt.load_epoch)
if os.path.exists(opt.save_dir) is False:
    os.makedirs(opt.save_dir)

for image, mask in iterator_test:
    iter_start_time = time.time()
    image = image.cuda()
    mask = mask.cuda()
    mask = mask[0][0]
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 1)
    mask = mask.byte()
    model.set_input(image, mask)
    model.set_gt_latent()
    model.test()
    real_A, real_B, fake_B = model.get_current_visuals()
    img = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0

    save_name = os.listdir(opt.test_srcdata)
    name_ = save_name[0]
    name = name_[:-4].strip()
    torchvision.utils.save_image(img, '%s/%s_%d.jpg' % (opt.save_dir, name, 1), normalize=True)
    torchvision.utils.save_image(img, '%s/%s_%d.jpg' % (opt.save_dir, name, 2), padding=0, normalize=True)
    torchvision.utils.save_image(real_A, '%s/%s_gmask.jpg' % (opt.save_dir, name), normalize=True)
    torchvision.utils.save_image(real_B, '%s/%s_gt.jpg' % (opt.save_dir, name), normalize=True)
    torchvision.utils.save_image(fake_B, '%s/%s_recs.jpg' % (opt.save_dir, name), normalize=True)
