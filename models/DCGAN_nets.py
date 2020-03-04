import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch

#CIFAR10
class DCGANGenerator_cifar10(nn.Module):
    def __init__(self, z_dim, ngf=64, output_nc=3,  norm_layer=nn.BatchNorm2d):
        super(DCGANGenerator_cifar10, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        seq = [nn.ConvTranspose2d(z_dim, ngf*8, 4, stride=1, padding=0, bias=use_bias),
               norm_layer(ngf*8),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf*8, ngf*4, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ngf*4),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ngf*2),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ngf),
               nn.ReLU(),
               nn.ConvTranspose2d(ngf, output_nc, 3, stride=1, padding=(1,1)),
               nn.Tanh()]

        self.model = nn.Sequential(*seq)

    def forward(self, input):
        return self.model(input.view(-1, self.z_dim, 1, 1))

class DCGANDiscriminator_cifar10(nn.Module):
    def __init__(self, ndf=64, input_nc=3, norm_layer=nn.BatchNorm2d):
        super(DCGANDiscriminator_cifar10, self).__init__()

        self.ndf = ndf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        seq = [nn.Conv2d(input_nc, ndf, 3, stride=1, padding=(1,1), bias=use_bias),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ndf*2),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ndf*4),
               nn.LeakyReLU(0.2),
               nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=(1,1), bias=use_bias),
               norm_layer(ndf*8),
               nn.LeakyReLU(0.2)]
        
        self.cnn_model = nn.Sequential(*seq)

        fc = [nn.Linear(4*4*ndf*8, 1)]
        self.fc = nn.Sequential(*fc)

    def forward(self, input):
        x = self.cnn_model(input)
        x = x.view(-1, 4*4*self.ndf*8)
        x = self.fc(x)
        return(x)

class DCGANGenerator_mnist(nn.Module):
    def __init__(self, z_dim, ngf=64, output_nc=1,  norm_layer=nn.BatchNorm2d):
        super(DCGANGenerator_mnist, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf
        self.img_size = 28*28*output_nc

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.fc1 = nn.Linear(self.z_dim, 256, bias=use_bias)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2, bias=use_bias)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2, bias=use_bias)
        self.fc4 = nn.Linear(self.fc3.out_features, self.img_size, bias=use_bias)
        self.model = nn.Sequential(
                self.fc1, 
                nn.LeakyReLU(negative_slope=0.2),
                self.fc2, 
                nn.LeakyReLU(negative_slope=0.2),
                self.fc3, 
                nn.LeakyReLU(negative_slope=0.2),
                self.fc4, 
                nn.Tanh()
        )


    def forward(self, noise):
        img = self.model(noise.view(noise.shape[0],-1))
        return img.view(noise.shape[0],1,28,28)

class DCGANDiscriminator_mnist(nn.Module):
    def __init__(self, ngf=64, input_nc=1,  norm_layer=nn.BatchNorm2d):
        super(DCGANDiscriminator_mnist, self).__init__()
        self.img_size = 28*28*input_nc
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.fc1 = nn.Linear(self.img_size, 1024, bias=use_bias)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2, bias=use_bias)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2, bias=use_bias)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.model = nn.Sequential(
                self.fc1, 
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.3),
                self.fc2, 
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.3),
                self.fc3, 
                nn.LeakyReLU(negative_slope=0.2),
                nn.Dropout(p=0.3),
                self.fc4, 
                nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img.view(img.shape[0], -1))
        return out

#TOY
class DCGANGenerator_toy(nn.Module):
    def __init__(self, z_dim, output=2, nd=512):
        super(DCGANGenerator_toy, self).__init__()
        #nosize size
        self.z_dim = z_dim
        seq = [
            nn.Linear(in_features=z_dim, out_features=nd),
            nn.ReLU(),
            nn.Linear(in_features=nd, out_features=nd),
            nn.ReLU(),
            nn.Linear(in_features=nd, out_features=nd),
            nn.ReLU(),
            nn.Linear(in_features=nd, out_features=2),
        ]

        self.model = nn.Sequential(*seq)

    def forward(self, input):
        #print("forward G(input)",input.size())
        return self.model(input.view(-1, self.z_dim))

class DCGANDiscriminator_toy(nn.Module):
    def __init__(self, output=2, nd=512):
        super(DCGANDiscriminator_toy, self).__init__()

        seq = [
                nn.Linear(in_features=2, out_features=nd),
                nn.ReLU(),
                nn.Linear(in_features=nd, out_features=nd),
                nn.ReLU(),
                nn.Linear(in_features=nd, out_features=nd),
                nn.ReLU(),
                nn.Linear(in_features=nd, out_features=1),
                nn.Sigmoid(),
            ]
        self.fc = nn.Sequential(*seq)

    def forward(self, input):
        x = self.fc(input)
        return(x)

