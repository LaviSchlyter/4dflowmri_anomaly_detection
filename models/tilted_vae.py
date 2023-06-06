import torch
import torch.nn as nn
import numpy as np

# this structure is based on https://arxiv.org/pdf/1810.01392.pdf
#https://github.com/anonconfsubaccount/tilted_prior/blob/main/train.py#L145
# to use for fair comparisons 

class Encoder(nn.Module):
    def __init__(self, shape, nz, mu_star, base):
        super(Encoder, self).__init__()
        self.mu_star = mu_star
        self.nz = nz
        if shape[0] == 28: # img size 28x28
            c = 2 # used for fully connected layer build
        elif shape[0] == 32: # img size 32x32
            c = 8

        self.conv = nn.Sequential(
            nn.Conv3d(shape[-1], base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(base, base, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(base, 2*base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(2*base, 2*base, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            #nn.Conv3d(2*base, 2*nz, kernel_size=7, stride=1, padding=0),
            nn.Conv3d(2*base, 2*nz, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU()
        )

        self.lin1 = nn.Linear(c*c*nz, nz)
        if mu_star == None:
            self.lin2 = nn.Linear(c*c*nz, nz)

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x, ood=False):
        x = self.conv(x)
        
        s = x.shape
        x = torch.reshape(x, (s[0], np.prod(s[1:])))

        mu = self.lin1(x)
        if self.mu_star == None: 
            logvar = self.lin2(x)
        else:
            logvar = torch.zeros_like(mu)
        
        z = self.reparametrize(mu,logvar)
        return [z, mu, logvar]
 
class Decoder(nn.Module):
    def __init__(self, shape, nz, loss_type, base=32):
        super(Decoder, self).__init__()
        self.shape = shape
        self.loss_type = loss_type 

        # for final convolutions factor
        if loss_type == 'l2':
            f = 1
        elif loss_type == 'cross_entropy':
            f = 256
        else:
            raise ValueError('{} is not a valid loss funtion, choose either l2 or cross_entropy'. format(loss_type))

        self.deconv = nn.Sequential( # nz
            nn.ConvTranspose3d(nz, 2*base, kernel_size=(shape[0]//4, shape[1]//4, shape[2]//4), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(2*base, 2*base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(2*base, 2*base, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(2*base, base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(base, base, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(base, base, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(base, f*shape[-1], kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        x = x.reshape((x.shape[1], x.shape[0], 1, 1))
        
        x = self.deconv(x)
        if self.loss_type == 'l2':
            return x
        else:
            x = x.view(-1, self.shape[2], 256, self.shape[0], self.shape[1])
            x = x.permute(0, 1, 3, 4, 2)
            return x

class TiltedVAE(nn.Module):
    def __init__(self, shape, nz, mu_star=None, loss_type='l2', base=32):
        super(TiltedVAE, self).__init__()
        self.encoder = Encoder(shape, nz, mu_star, base = base)
        self.decoder = Decoder(shape, nz, loss_type, base = base)

        # Initialize encoder and decoder weights
        self.initialize_weights()

        self.shape = shape
        self.nz = nz
        self.loss_type = loss_type
    
    def forward(self, x, ood=False):
        z, mu, logvar = self.encoder(x, ood=ood)
        x = self.decoder(z)
        dict = {'decoder_output': x, 'encoder_z': z, 'mu': mu, 'logvar': logvar}
        return dict
    
    def initialize_weights(self):
        for m in self.encoder.modules():
            self.weights_init(m)

        for m in self.decoder.modules():
            self.weights_init(m)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
    
