# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:03:07 2021

@author: hp
"""
import os, time, pickle, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from util import *
from network import *
from dataloader import *

opt = {'dataset': 'facades', 'train_subfolder': 'train', 'test_subfolder': 'val', 'batch_size': 4, 'test_batch_size': 5,
 'ngf': 64, 'ndf': 64, 'input_size': 256, 'crop_size': 256, 'resize_scale': 286, 'fliplr': True, 'train_epoch': 200,
 'lrD': .0002, 'lrG': .0002, 'L1_lambda': 100, 'beta1': .5, 'beta2': .999, 'save_root': 'results', 'inverse_order': False}

# results save path
main_dir = 'dataset_demo/FLIR_ADAS_1_3/'
root = os.getcwd() + '/' + opt['save_root'] + '/'
model = 'flir' + '_'


if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

# data_loader
# transform_rgb = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#         transforms.Resize((256,256)),
# ])

# transform_ir = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5), std=(0.5)),
#         transforms.Resize((256,256)),
# ])

train_dataset = Dataset(main_dir + 'train', 256)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt['batch_size'],
                                           shuffle = True, num_workers = 1)

test_dataset = Dataset(main_dir + 'val', 256)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt['test_batch_size'],
                                          shuffle = True, num_workers = 0)



test = test_loader.__iter__().__next__()
#print("test size: ", test.size())

fixed_y_ = test[1]
fixed_x_ = test[0]

print("fixed_x_ size: ", fixed_x_.size())
print("fixed_y_ size: ", fixed_x_.size())

#img_size = test.size()[2]
# if opt['inverse_order']:
#     fixed_y_ = test[:, :, :, 0:img_size]
#     fixed_x_ = test[:, :, :, img_size:]
#     #print("fixed_x_ size: ", fixed_x_.size())
# else:
#     fixed_x_ = test[:, :, :, 0:img_size]
#     fixed_y_ = test[:, :, :, img_size:]

# if img_size != opt['input_size']:
#     fixed_x_ = imgs_resize(fixed_x_, opt['input_size'])
#     fixed_y_ = imgs_resize(fixed_y_, opt['input_size'])

# network
G = generator(opt['ngf'])
D = discriminator(opt['ndf'])
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G.train()
D.train()

# loss
BCE_loss = nn.BCELoss().cuda()
L1_loss = nn.L1Loss().cuda()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=opt['lrG'], betas=(opt['beta1'], opt['beta2']))
D_optimizer = optim.Adam(D.parameters(), lr=opt['lrD'], betas=(opt['beta1'], opt['beta2']))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in tqdm(range(opt['train_epoch'])):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    for x_, y_ in tqdm(train_loader):
        #print(x_.size(), y_.size())
        # train discriminator D
        D.zero_grad()

        # if opt['inverse_order']:
        #     y_ = x_[:, :, :, 0:img_size]
        #     x_ = x_[:, :, :, img_size:]
        # else:
        #     y_ = x_[:, :, :, img_size:]
        #     x_ = x_[:, :, :, 0:img_size]
            
        # if img_size != opt['input_size']:
        #     x_ = imgs_resize(x_, opt['input_size'])
        #     y_ = imgs_resize(y_, opt['input_size'])

        # if opt['resize_scale']:
        #     x_ = imgs_resize(x_, opt['resize_scale'])
        #     y_ = imgs_resize(y_, opt['resize_scale'])

        # if opt['crop_size']:
        #     x_, y_ = random_crop(x_, y_, opt['crop_size'])

        # if opt['fliplr']:
        #     x_, y_ = random_fliplr(x_, y_)
        
        # print(x_.size(), y_.size())

        x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())

        D_result = D(x_, y_).squeeze()
        D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda()))

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda()))

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        D_optimizer.step()

        train_hist['D_losses'].append(D_train_loss.item())

        D_losses.append(D_train_loss.item())

        # train generator G
        G.zero_grad()

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()

        G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + opt['L1_lambda'] * L1_loss(G_result, y_)
        G_train_loss.backward()
        G_optimizer.step()

        train_hist['G_losses'].append(G_train_loss.item())

        G_losses.append(G_train_loss.item())

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    avg_psnr = average_psnr(G, x_, y_)
    #print("Average PSNR: ", avg_psnr)

    avg_ssim = average_ssim(G, x_, y_)
    #print("Average SSIM: ", avg_ssim)

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f, Average PSNR: %.4f, Average SSIM: %.4f' % ((epoch + 1), opt['train_epoch'], per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses)), avg_psnr, avg_ssim))
    #fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    if (epoch+1) % 10. == 0.:
      print("Saving Generated Images...")
      fixed_p = root + 'Fixed_results/Epoch_wise_result/' + str(epoch + 1) + '/'

      if not os.path.exists(fixed_p):
        os.makedirs(fixed_p)

      # show_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, (epoch+1), save=True, path=fixed_p)
      show_individual_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, save=True, path=fixed_p)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')