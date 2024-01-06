import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torch.optim as optim
import matplotlib.pyplot as plt

class WSConv2d(nn.Module):
    """
    Weighted Scale Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.scale = (2 / (in_channels * kernel_size[0]**2 ))**0.5

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.bias = Parameter(torch.Tensor(out_channels))

        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.conv2d(input=x, weight=self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)


class PixelNormalization(nn.Module):
    """
        Performs pixel normalization in each channel
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # N x C x H x W
        denominator = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        return x / denominator


class Minibatch_std(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1

        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)

class FromRGB(nn.Module):
    """
    Input image through weighted scale convolution
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: number of input channels in conv.
        :param out_channels: number of output channels in conv.
        """
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
        self.lRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.lRelu(self.conv(x))


class ToRGB(nn.Module):
    """
    Convert given image to RGB image with weighted scale convolution - kernel_size(1,1), stride=(1,1).
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: number of input channels in conv.
        :param out_channels: number of output channels in conv.
        """
        super().__init__()
        self.conv = WSConv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.conv(x)


class ConvGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial_block=False):
        super().__init__()
        if initial_block:
            self.upsample = None
            self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))  # pad=3 since 1x1 -> 4x4

        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.leakyR = nn.LeakyReLU(0.2)
        self.pn = PixelNormalization()
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        # first convolution and pixel normalization
        x = self.leakyR(self.conv1(x))
        x = self.pn(x)

        # second convolution and pixel normalization
        x = self.leakyR(self.conv2(x))
        x = self.pn(x)
        return x


class ConvDBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initial_block=None):
        super().__init__()

        if initial_block:
            self.miniBatchStd = Minibatch_std()
            self.conv1 = WSConv2d(in_channels + 1, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=(4, 4),
                                  stride=(1, 1))  # in_channels=out_channels , pad=0,=> 4x4 -> 1x1
            self.outLayer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_channels, 1)
            )

        else:
            self.miniBatchStd = None
            self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.conv2 = WSConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.outLayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # downsampling with avgpooling

        self.leakyR = nn.LeakyReLU(0.2)
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.miniBatchStd is not None:
            x = self.miniBatchStd(x)

        # first convolution and leakyRelu
        x = self.leakyR(self.conv1(x))

        # second convolution and leakyRelu
        x = self.leakyR(self.conv2(x))

        # output layer
        x = self.outLayer(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, out_res):
        super().__init__()
        # initially
        self.depth = 1
        self.alpha = 1
        self.fade_iters = 0
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.current_layers = nn.ModuleList([ConvGBlock(z_dim, z_dim, initial_block=True)])
        self.rgbs_layers = nn.ModuleList([ToRGB(z_dim, 1)])

        for layer in range(2, int(np.log2(out_res))):  # np.log2(256) = 8
            if layer < 6:
                # All low resolution blocks 8x8, 16x16, 32x32 with same 512 channels
                in_channels, out_channels = 512, 512

            else:
                # layer > 6 : 5th block(64x64), the number of channels halved for each block
                in_channels, out_channels = int(512 / 2 ** (layer - 6)), int(512 / 2 ** (layer - 6))

            self.current_layers.append(ConvGBlock(in_channels, out_channels))
            self.rgbs_layers.append(ToRGB(out_channels, 1))

    def forward(self, x):
        for block in self.current_layers[:self.depth - 1]:
            x = block(x)

        out = self.current_layers[self.depth - 1](x)

        x_rgb_out = self.rgbs_layers[self.depth - 1](out)
        if self.alpha < 1:
            x_old = self.upsample(x)
            old_rgb = self.rgbs_layers[self.depth - 2](x_old)
            x_rgb_out = (1 - self.alpha) * old_rgb + self.alpha * x_rgb_out

            self.alpha += self.fade_iters

        return x_rgb_out

    def growing_net(self, num_iters):
        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters

        self.depth += 1


class Discriminator(nn.Module):
    def __init__(self, z_dim, out_res):
        super().__init__()
        # initially
        self.depth = 1
        self.alpha = 1  # between 0 to 1, increasing later on
        self.fade_iters = 0
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.current_layers = nn.ModuleList([ConvDBlock(z_dim, z_dim, initial_block=True)])
        self.fromRgbLayers = nn.ModuleList([FromRGB(1, z_dim)])

        for layer in range(2, int(np.log2(out_res))):  # np.log2(256) = 8
            if layer < 6:
                # All low resolution blocks 8x8, 16x16, 32x32 with same 512 channels
                in_channels, out_channels = 512, 512

            else:
                # layer > 6 : 5th block(64x64), the number of channels halved for each block
                in_channels, out_channels = int(512 / 2 ** (layer - 6)), int(512 / 2 ** (layer - 6))

            self.current_layers.append(ConvDBlock(in_channels, out_channels))
            self.fromRgbLayers.append(FromRGB(1, in_channels))

    def forward(self, x_rgb):
        x = self.fromRgbLayers[self.depth - 1](x_rgb)

        x = self.current_layers[self.depth - 1](x)

        if self.alpha < 1:
            x_rgb = self.downsample(x_rgb)
            x_old = self.fromRgbLayers[self.depth - 2](x_rgb)
            x = (1 - self.alpha) * x_old + self.alpha * x
            self.alpha += self.fade_iters

        for block in reversed(self.current_layers[:self.depth - 1]):
            x = block(x)

        return x

    def growing_net(self, num_iters):
        self.fade_iters = 1 / num_iters
        self.alpha = 1 / num_iters

        self.depth += 1


root = '../'
data_dir = '../datasets/'
check_point_dir = './progan/check_points/'
output_dir = './progan/output/'
weight_dir = './progan/weight/'

if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)

schedule = [[5, 15, 25, 35, 40, 45], [16, 16, 16, 8, 4, 4], [5, 5, 5, 1, 1, 1]]
batch_size = schedule[1][0]
growing = schedule[2][0]
epochs = 45
latent_size = 512
out_res = 128
lr = 1e-4
lambd = 10

tmp = False

device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

transform = transforms.Compose([
    transforms.Resize(out_res),
    transforms.CenterCrop(out_res),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

D_net = Discriminator(latent_size, out_res).to(device)
G_net = Generator(latent_size, out_res).to(device)

fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
D_optimizer = optim.Adam(D_net.parameters(), lr=lr, betas=(0.5, 0.99))
G_optimizer = optim.Adam(G_net.parameters(), lr=lr, betas=(0.5, 0.99))

D_running_loss = 0.0
G_running_loss = 0.0
iter_num = 0

D_epoch_losses = []
G_epoch_losses = []

if tmp:
    check_point = torch.load(check_point_dir + 'check_point_epoch_%i.pth' % opt.resume)
    fixed_noise = check_point['fixed_noise']
    G_net.load_state_dict(check_point['G_net'])
    D_net.load_state_dict(check_point['D_net'])
    G_optimizer.load_state_dict(check_point['G_optimizer'])
    D_optimizer.load_state_dict(check_point['D_optimizer'])
    G_epoch_losses = check_point['G_epoch_losses']
    D_epoch_losses = check_point['D_epoch_losses']
    G_net.depth = check_point['depth']
    D_net.depth = check_point['depth']
    G_net.alpha = check_point['alpha']
    D_net.alpha = check_point['alpha']

try:
    c = next(x[0] for x in enumerate(schedule[0]) if x[1] > 0) - 1
    batch_size = schedule[1][c]
    growing = schedule[2][c]
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    tot_iter_num = (len(dataset) / batch_size)
    G_net.fade_iters = (1 - G_net.alpha) / (schedule[0][c + 1] - 0) / (2 * tot_iter_num)
    D_net.fade_iters = (1 - D_net.alpha) / (schedule[0][c + 1] - 0) / (2 * tot_iter_num)


except:
    print('Fully Grown\n')
    c = -1
    batch_size = schedule[1][c]
    growing = schedule[2][c]

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    tot_iter_num = (len(dataset) / batch_size)
    print(schedule[0][c], 0)

    if G_net.alpha < 1:
        G_net.fade_iters = (1 - G_net.alpha) / (epochs - 0) / (2 * tot_iter_num)
        D_net.fade_iters = (1 - D_net.alpha) / (epochs - 0) / (2 * tot_iter_num)

size = 2 ** (G_net.depth + 1)
print("Output Resolution: %d x %d" % (size, size))

for epoch in range(1, epochs + 1):
    G_net.train()
    D_epoch_loss = 0.0
    G_epoch_loss = 0.0
    if epoch - 1 in schedule[0]:

        if (2 ** (G_net.depth + 1) < out_res):
            c = schedule[0].index(epoch - 1)
            batch_size = schedule[1][c]
            growing = schedule[2][0]
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            tot_iter_num = (len(dataset) / batch_size)
            G_net.growing_net(growing * tot_iter_num)
            D_net.growing_net(growing * tot_iter_num)
            size = 2 ** (G_net.depth + 1)
            print("Output Resolution: %d x %d" % (size, size))

    print("epoch: %i/%i" % (int(epoch), int(epochs)))
    databar = tqdm(data_loader)

    for i, samples in enumerate(databar):
        ##  update D
        if size != out_res:

            samples = F.interpolate(samples[0], size=size).to(device)
        else:
            samples = samples[0].to(device)
        D_net.zero_grad()
        noise = torch.randn(samples.size(0), latent_size, 1, 1, device=device)
        fake = G_net(noise)
        fake_out = D_net(fake.detach())
        real_out = D_net(samples)

        ## Gradient Penalty

        eps = torch.rand(samples.size(0), 1, 1, 1, device=device)
        eps = eps.expand_as(samples)
        x_hat = eps * samples + (1 - eps) * fake.detach()
        x_hat.requires_grad = True
        px_hat = D_net(x_hat)
        grad = torch.autograd.grad(
            outputs=px_hat.sum(),
            inputs=x_hat,
            create_graph=True
        )[0]
        grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
        gradient_penalty = lambd * ((grad_norm - 1) ** 2).mean()

        ###########

        D_loss = fake_out.mean() - real_out.mean() + gradient_penalty

        D_loss.backward()
        D_optimizer.step()

        ##	update G

        G_net.zero_grad()
        fake_out = D_net(fake)

        G_loss = - fake_out.mean()

        G_loss.backward()
        G_optimizer.step()

        ##############

        D_running_loss += D_loss.item()
        G_running_loss += G_loss.item()

        iter_num += 1

        if i % 500 == 0:
            D_running_loss /= iter_num
            G_running_loss /= iter_num
            print('iteration : %d, gp: %.2f' % (i, gradient_penalty))
            databar.set_description('D_loss: %.3f   G_loss: %.3f' % (D_running_loss, G_running_loss))
            iter_num = 0
            D_running_loss = 0.0
            G_running_loss = 0.0

    D_epoch_losses.append(D_epoch_loss / tot_iter_num)
    G_epoch_losses.append(G_epoch_loss / tot_iter_num)

    with torch.no_grad():
        G_net.eval()
        torch.save(G_net.state_dict(), weight_dir + 'G_weight_epoch_%d.pth' % (epoch))
        out_imgs = G_net(fixed_noise)
        out_grid = make_grid(out_imgs, normalize=True, nrow=4, scale_each=True,
                             padding=int(0.5 * (2 ** G_net.depth))).permute(1, 2, 0)
        plt.imshow(out_grid.cpu())
        plt.savefig(output_dir + 'size_%i_epoch_%d' % (size, epoch))

