import time
from models.utils import Data_load
from models.models import create_model
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', type=str, default='.\Dataset\Partial_dataset',
                    help='The file storing the names of the file for training (If not provided training '
                         'will happen for all images in train_dir)')
parser.add_argument('--mask_dataset', type=str, default='.\Dataset\Mask_dataset',
                    help='The file storing the names of the file for training (If not provided training '
                         'will happen for all images in mask_dir)')
parser.add_argument('--checkpoints_dir', type=str, default=r'.\checkpoints', help="here models are saved "
                                                                                  "during training")
parser.add_argument('--save_dir', type=str, default='./save', help='Training Result Save Path')
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
parser.add_argument('--lambda_A', type=int, default=100)
parser.add_argument('--threshold', type=float, default=5 / 16.0)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--shift_sz', type=int, default=1, help="size of feature patch")
parser.add_argument('--mask_thred', type=int, default=1)
parser.add_argument('--strength', type=int, default=1)
parser.add_argument('--init_gain', type=float, default=0.02)
parser.add_argument('--cosis', type=int, default=1)
parser.add_argument('--gan_type', type=str, default='lsgan')
parser.add_argument('--gan_weight', type=int, default=0.2)
parser.add_argument('--overlap', type=int, default=4)
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--display_freq', type=int, default=1000)
parser.add_argument('--save_epoch_freq', type=int, default=2, help="Save training parameters")
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--epoch_count', type=int, default=1)
parser.add_argument('--which_epoch', type=str, default='')
parser.add_argument('--niter', type=int, default=1)
parser.add_argument('--niter_decay', type=int, default=5)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_policy', type=str, default='lambda')
parser.add_argument('--lr_decay_iters', type=int, default=50)
parser.add_argument('--isTrain', type=bool, default=True)

opt = parser.parse_args()
transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(), ])
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
dataset_train = Data_load(opt.train_dataset, opt.mask_dataset, transform, transform_mask)
iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True))
print(len(dataset_train))

model = create_model(opt)
total_steps = 0
iter_start_time = time.time()
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for image, mask in iterator_train:
        image = image.cuda()
        mask = mask.cuda()
        mask = mask[0][0]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(image, mask)
        model.set_gt_latent()
        model.optimize_parameters()
        if total_steps % opt.display_freq == 0:
            real_A, real_B, fake_B = model.get_current_visuals()
            img = (torch.cat([real_A, real_B, fake_B], dim=0) + 1) / 2.0
            torchvision.utils.save_image(img, '%s/Epoch%d_%d of %d.jpg' % (
                opt.save_dir, epoch, total_steps + 1, len(dataset_train)), nrow=1)
        if total_steps % 1 == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print(errors)
    if epoch % opt.save_epoch_freq == 0:
        print('Save at the end of the epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)
    print('Epoch end %d / %d \t Total Time: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
