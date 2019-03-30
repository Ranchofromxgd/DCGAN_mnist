from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
from PIL import Image
import math
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision
from torch.optim.lr_scheduler import StepLR
from corelation import corelation

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default= './data/training_data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=12, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, default=999,help='manual seed')
opt = parser.parse_args()
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

if (not os.path.exists(opt.dataroot)):
    os.mkdir(opt.dataroot)
if(not os.path.exists('./data/train_result')):
    os.mkdir('./data/train_result')

global learning_rate
learning_rate = opt.lr

#epoches in this list will lower learning rate 
schedule = list(range(1,200,10))
#Decreasing rate of learning rate
gamma = 0.8
#Gray image, so only one channel
nc=1
#Mnist , so size is 28*28*1
classifier_input_imagesize = 28
#store the losses
loss_dict = {} 
#Pretrained classifier
model_dict = './data/pretrained_classifier/CNN_mnist.pth'


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))

assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# custom weights initialization called on netG and netDd
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if self.ngpu > 1 and input.is_cuda:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
	    nn.Conv2d(
	        in_channels=1,              # input height
	        out_channels=16,            # n_filters
	        kernel_size=5,              # filter size
	        stride=1,                   # filter movement/step
	        padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
	    ),                              # output shape (16, 28, 28)
	    nn.ReLU(),                      # activation
	    nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
            )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
	    nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
	    nn.ReLU(),                      # activation
	    nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


if(opt.cuda):
    model = CNN().cuda()
else:
    model = CNN()

model.load_state_dict(torch.load(model_dict)) 

transform=transforms.Compose([
                                   transforms.Resize(classifier_input_imagesize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                               ])
                               
netG = Generator(ngpu).to(device)
netG.apply(weights_init)

#Input is reuslt of Generator
#output the classication result and similarity
def classify(G_result):
    vutils.save_image(G_result.detach(),
		       './data/temp.png',
			normalize=True,
			nrow = 1)
    img_test = np.array(Image.open('./data/temp.png'))
    ndarray_convert_img= Image.fromarray(G_result[0,0,:,:].cpu().detach().numpy())
    pic = transform(ndarray_convert_img)
    temp = torch.zeros(1,1,classifier_input_imagesize,classifier_input_imagesize)
    temp[0,0,:,:] = pic
    if(opt.cuda):
        temp = temp.cuda()
    result,_ = model(temp)     
    result = result[0].cpu().detach().numpy().tolist()
    index = result.index(max(result))  #The regonition result for the generated picture
    img_standard = np.array(Image.open('./data/standerd_pic/'+str(index)+'_standerd.png'))
    score = corelation(img_test,img_standard)
    return index,score
    
#Use this function to evaluate how good is our generator
def evaluate(k1 = 0,k2 = 0,k3 = 30,pic_num = 40,criteria = 0):

    angle = np.linspace(0,2*math.pi,pic_num)

    noise_x = np.sin(angle)
    noise_y = np.cos(angle)

    squence = []
    col_score = []

    def score(sq,cs): 
        num_classes = len(set(sq))
        num_divided  = 0
        now = sq[0]
        diff_col = sum(cs)-criteria
        for i in range(1,len(sq)):
	        if(sq[i] == sq[i-1]):
	            continue
	        else:
	            num_divided+=1
        if(diff_col < 0.1):
	        diff_col = 0.1  
        score = k1*(1/num_classes)+k2*(num_divided/num_classes)+k3*(1/num_classes)*pic_num/diff_col
        return score

    for each in zip(noise_x,noise_y):
        noise = torch.zeros(1,nz,1,1)
        noise[0,0,0,0] = each[0]  #We can try to switch these two coo to see what will happen
        noise[0,1,0,0] = each[1]
        if(opt.cuda):
            G_result = netG(noise.cuda())
        else:
            G_result = netG(noise)
        index,col = classify(G_result)
        col_score.append(col)
        squence.append(index)

    result = score(squence,col_score)

    return result

#Adjust GAN's learning rate by editing :schedule
def adjust_learning_rate(optimizer, epoch,change = False):
    global learning_rate
    if epoch in schedule:
        if(change == True):
            learning_rate *= gamma
	#We can adjust the learning rate through this loop
        for param_group in optimizer.param_groups:
            print(optimizer)
            print("learning rate adjust to :",learning_rate)
            param_group['lr'] = learning_rate

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
criterion = nn.BCELoss()

#IF WE USE UNIFORM DISTRIBUTION
#fixed_noise = torch.from_numpy(np.random.uniform(-1,1,(opt.batchSize,nz,1,1)))
#fixed_noise = fixed_noise.type(torch.FloatTensor)
#IF WE USE GUASSIAN DISTRIBUTION
fixed_noise = torch.randn(opt.batchSize,nz,1,1)

real_label = 1
fake_label = 0
cnt = 0
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(opt.beta1, 0.999))
print('Start training...')
for epoch in range(opt.niter):
    adjust_learning_rate(optimizerD,epoch,change = True)
    adjust_learning_rate(optimizerG,epoch)
    loss_dict[epoch] = {} #Every epoch have its own dict
    loss_dict[epoch]['loss_d'] = []
    loss_dict[epoch]['loss_g'] = []
    loss_dict[epoch]['loss_dx'] = []
    loss_dict[epoch]['loss_dgx1'] = []
    loss_dict[epoch]['loss_dgx2'] = []
    for i, data in enumerate(dataloader, 0):
        cnt += 1
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

#	IF WE USE UNIFORM DISTRIBUTION
#       noise = torch.from_numpy(np.random.uniform(-1,1,(batch_size,nz,1,1)).astype(np.float32))
#       noise = noise.type(torch.FloatTensor)
#	IF WE USE GUASSIAN DISTRIBUTION
        noise = torch.randn(batch_size, nz, 1, 1)

        if(opt.cuda):
            fake = netG(noise.cuda())
        else:
            fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        evalu = evaluate()
        errG = criterion(output, label)+0.01*evalu
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f \t Loss_G: %.4f \tOverall Generator loss: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(),evalu))
        loss_dict[epoch]['loss_d'].append(errD.item())
        loss_dict[epoch]['loss_g'].append(errG.item())
        loss_dict[epoch]['loss_dx'].append(D_x)
        loss_dict[epoch]['loss_dgx1'].append(D_G_z1)
        loss_dict[epoch]['loss_dgx2'].append(D_G_z2)
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % './data/train_result',
                    normalize=True)
            if(opt.cuda):
                fake = netG(fixed_noise.cuda())
            else:
                fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % ('./data/train_result', epoch),
                    normalize=True)

        # do checkpointing after several batches
        if(cnt %20 == 0 ):
            torch.save(netG.state_dict(), '%s/'%('./data/train_result')+str(cnt/20)+'_epoch'+str(epoch)+'netG_epoch_%d.pth' % ( epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ('./data/train_result', epoch))


import pickle
pickle_file = open('./losses.pkl','wb')
print('Saving losses params...')
pickle.dump(loss_dict,pickle_file)
pickle_file.close()
print('Done')

    
