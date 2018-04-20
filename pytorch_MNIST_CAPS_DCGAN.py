import os, time
import itertools
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from capsule_network import *
import argparse


USE_CUDA=torch.cuda.is_available()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=str, default='mnist', help='cifar10 | imagenet | folder | lfw ')
parser.add_argument('--caps_D', required=False, action='store_true', default=True, help='use caps as descriminator?')
parser.add_argument('--save_training', required=False, action='store_true', default=False, help='use caps as descriminator?')

parser.add_argument('--SN', action='store_true',default=True, help='Enables Spectral Normalization ')
opt = parser.parse_args()


# G(z)i

size=32

class generator(nn.Module):
    # initializers
    def __init__(self, d=128,img_size=size):
        super(generator, self).__init__()
        self.img_size=size
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        
        if self.img_size==64:
            self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
            self.deconv4_bn = nn.BatchNorm2d(d)
            self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        if self.img_size==32: 
            self.deconv4= nn.ConvTranspose2d(d*2,1,4,2,1)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        if self.img_size==64:
            x = F.relu(self.deconv4_bn(self.deconv4(x)))
            x = F.tanh(self.deconv5(x))
        if self.img_size==32:
            x= F.tanh(self.deconv4(x))
        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128,img_size=size):
        super(discriminator, self).__init__()
        self.img_size=img_size
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        
        if self.img_size==64:
            self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
            self.conv4_bn = nn.BatchNorm2d(d*8)
            self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)
        if self.img_size==32:
            self.conv4 = nn.Conv2d(d*4,1,4,1,0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        if self.img_size==64:
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = F.sigmoid(self.conv5(x))
        if self.img_size==32:
            x=F.sigmoid(self.conv4(x))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise

if USE_CUDA:
    fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
else:
    fixed_z_ = Variable(fixed_z_, volatile=True)


def save_result(path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)

    if USE_CUDA:
        z_ = Variable(z_.cuda(), volatile=True)
    else:
        z_ = Variable(z_, volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    vutils.save_image(test_images.data,path)




# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = size
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
if opt.caps_D:
    D=CapsNet() #already initlialized
else:
    D = discriminator(d=128)
    D.weight_init(mean=0.0, std=0.02)

#generator
G = generator(d=128)
G.weight_init(mean=0.0, std=0.02)


if USE_CUDA:
    G.cuda()
    D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNIST_DCGAN_results'):
    os.mkdir('MNIST_DCGAN_results')
if not os.path.isdir('MNIST_DCGAN_results/Random_results'):
    os.mkdir('MNIST_DCGAN_results/Random_results')
if not os.path.isdir('MNIST_DCGAN_results/Fixed_results'):
    os.mkdir('MNIST_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        if USE_CUDA:
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        else:
            x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)

        if USE_CUDA:
            z_ = Variable(z_.cuda())
        else:
            z_ = Variable(z_)

        G_result = G(z_)

        D_result = D(G_result).squeeze()

        D_fake_loss = BCE_loss(D_result, y_fake_)

        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # D_losses.append(D_train_loss.data[0])
        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)

        if USE_CUDA:
            z_ = Variable(z_.cuda())
        else:
            z_ = Variable(z_)

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])
        num_iter += 1
        if num_iter%1==0:
            print('epoch: [%d/%d] batch: [%d] loss_d: %.3f loss_g: %.3f' %  (epoch+1,train_epoch,num_iter,D_train_loss.data[0],G_train_loss.data[0]))
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = 'MNIST_DCGAN_results/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '_size_'+str(size)+'.png'
    fixed_p = 'MNIST_DCGAN_results/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '_size_'+str(size) + '.png'
    save_result(fixed_p,isFix=True)
    save_result(p,isFix=False)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")

if opt.save_training:
    torch.save(G.state_dict(), "MNIST_DCGAN_results/generator_param.pkl")
    torch.save(D.state_dict(), "MNIST_DCGAN_results/discriminator_param.pkl")
    with open('MNIST_DCGAN_results/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)
