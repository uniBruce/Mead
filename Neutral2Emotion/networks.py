import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Audio2Exp(nn.Module):
    def __init__(self, params):
        # initialize and input & output dimension
        super(Audio2Exp, self).__init__()
        self.input_dim = params['audio_dim']
        self.output_dim = params['exp_dim']
        self.init_hidden(params)
        # construct model
        self.lstm = nn.LSTM(self.input_dim, params['lstm']['node_dim'], params['lstm']['layer_num'], batch_first=True)
        self.fc = nn.Linear(params['lstm']['node_dim'], self.output_dim)

    def init_hidden(self, params):
        # Create the trainable initial state
        h_init = init.constant_(torch.empty(params['lstm']['layer_num'], params['batch_size'], params['lstm']['node_dim'], dtype=torch.float32), 0.0)
        c_init = init.constant_(torch.empty(params['lstm']['layer_num'], params['batch_size'], params['lstm']['node_dim'], dtype=torch.float32), 0.0)
        self.hidden = (h_init.cuda(), c_init.cuda())
        self.hidden = self.hidden

    def forward(self, audio):
        output, _ = self.lstm(audio, self.hidden)
        output = self.fc(output)
        return output[:,-1]


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        #relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        #relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        #relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return [relu3_3, relu5_3]

class Unet_Enc_384x384(nn.Module):
    def __init__(self, nc_input_x = 3):
        super(Unet_Enc_384x384, self).__init__()

        self.struc_enc_pre_ = enc_pre_block_v9(nc_input_x, 32)
        self.struc_enc_32_64_ = enc_block_v9(32, 64)
        self.struc_enc_64_128_ = enc_block_v9(64, 128)
        self.struc_enc_128_256_ = enc_block_v9(128, 256)
        self.struc_enc_256_512 = enc_block_v9(256, 512)
        self.struc_enc_512_512_0 = enc_block_v9(512, 512)
        self.struc_enc_512_512_1 = enc_block_v9(512, 512)
        self.struc_enc_inner_most_ = enc_block_v9(512, 8)

    def forward(self, x):

        ## structure-to-image unet pass
        # endoer
        y0 = self.struc_enc_pre_(x) # size: (n,32,192,192)
        y1 = self.struc_enc_32_64_(y0) # size: (n,64,96,96)
        y2 = self.struc_enc_64_128_(y1) # size: (n,128,48,48)
        y3 = self.struc_enc_128_256_(y2) # size: (n,256,24,24)
        y4 = self.struc_enc_256_512(y3) # size: (n,512,12,12)
        y5 = self.struc_enc_512_512_0(y4) # size: (n,512,6,6)
        y6 = self.struc_enc_512_512_1(y5) # size: (n,512,3,3)
        y7 = self.struc_enc_inner_most_(y6) # size: (n,7,1,1)

        embedding = y7.squeeze(3).squeeze(2)


        return embedding


class Unet_Int_384x384(nn.Module):
    def __init__(self, nc_input_x = 3):
        super(Unet_Int_384x384, self).__init__()

        self.struc_enc_pre_ = enc_pre_block_v9(nc_input_x, 32)
        self.struc_enc_32_64_ = enc_block_v9(32, 64)
        self.struc_enc_64_128_ = enc_block_v9(64, 128)
        self.struc_enc_128_256_ = enc_block_v9(128, 256)
        self.struc_enc_256_512 = enc_block_v9(256, 512)
        self.struc_enc_512_512_0 = enc_block_v9(512, 512)
        self.struc_enc_512_512_1 = enc_block_v9(512, 512)
        self.struc_enc_inner_most_ = enc_block_v9(512, 3)

    def forward(self, x):

        ## structure-to-image unet pass
        # endoer
        y0 = self.struc_enc_pre_(x) # size: (n,32,192,192)
        y1 = self.struc_enc_32_64_(y0) # size: (n,64,96,96)
        y2 = self.struc_enc_64_128_(y1) # size: (n,128,48,48)
        y3 = self.struc_enc_128_256_(y2) # size: (n,256,24,24)
        y4 = self.struc_enc_256_512(y3) # size: (n,512,12,12)
        y5 = self.struc_enc_512_512_0(y4) # size: (n,512,6,6)
        y6 = self.struc_enc_512_512_1(y5) # size: (n,512,3,3)
        y7 = self.struc_enc_inner_most_(y6) # size: (n,3,1,1)

        embedding = y7.squeeze(3).squeeze(2)


        return embedding


class Unet_Dec_384x384(nn.Module):
    def __init__(self,  nc_output =3):
        super(Unet_Dec_384x384, self).__init__()
        self.struc_dec_inner_most_ = dec_block_v9(512, 512, use_dropout=False, pixel_shuffle=False)
        self.struc_dec_1024_512_1_ = dec_block_v9(1024, 512, use_dropout=False, pixel_shuffle=False)
        self.struc_dec_1024_512_2_ = dec_block_v9(1024, 512, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_1024_256_ = dec_block_v9(1024, 256, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_512_128_ = dec_block_v9(512, 128, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_256_128_ = dec_block_v9(256, 64, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_256_64_ = dec_block_v9(128, 64, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_post_ = dec_post_block_v9(128, nc_output, pixel_shuffle=True, act_post=False)

    def forward(self, y):

        ## structure-to-image unet pass
        y0 = y[0]
        y1 = y[1]
        y2 = y[2]
        y3 = y[3]
        y4 = y[4]
        y5 = y[5]
        y6 = y[6]
        y7 = y[7]

        unet_dec_y = self.struc_dec_inner_most_(y7) # size: (n,512,3,3)
        unet_dec_y = torch.cat([unet_dec_y, y6], 1) # size: (n,1024,3,3)
        unet_dec_y = self.struc_dec_1024_512_1_(unet_dec_y)  # size: (n,512,6,6)
        unet_dec_y = torch.cat([unet_dec_y, y5], 1) # size: (n,1024,6,6)
        unet_dec_y = self.struc_dec_1024_512_2_(unet_dec_y)  # size: (n,512,12,12)
        unet_dec_y = torch.cat([unet_dec_y, y4], 1) # size: (n,1024,12,12)
        unet_dec_y = self.struc_dec_1024_256_(unet_dec_y)  # size: (n,256,24,24)
        unet_dec_y = torch.cat([unet_dec_y, y3], 1) # size: (n,512,24,24)
        unet_dec_y = self.struc_dec_512_128_(unet_dec_y)  # size: (n,128,48,48)
        unet_dec_y = torch.cat([unet_dec_y, y2], 1) # size: (n,256,48,48)
        unet_dec_y = self.struc_dec_256_128_(unet_dec_y)  # size: (n,128,96,96)
        unet_dec_y = torch.cat([unet_dec_y, y1], 1) # size: (n,256,96,96)
        unet_dec_y = self.struc_dec_256_64_(unet_dec_y)  # size: (n,64,192,192)
        unet_dec_y = torch.cat([unet_dec_y, y0], 1)  # size: (n,128,192,192)
        unet_dec_y = self.struc_dec_post_(unet_dec_y)  # size: (n,3,384,384)
        output = unet_dec_y

        return output


class Unet_V9_384x384(nn.Module):
    def __init__(self, nc_input_x = 4, nc_output=3):
        super(Unet_V9_384x384, self).__init__()

        self.struc_enc_pre_ = enc_pre_block_v9(nc_input_x, 32)
        self.struc_enc_32_64_ = enc_block_v9(32, 64)
        self.struc_enc_64_128_ = enc_block_v9(64, 128)
        self.struc_enc_128_256_ = enc_block_v9(128, 256)
        self.struc_enc_256_512 = enc_block_v9(256, 512)
        self.struc_enc_512_512_0 = enc_block_v9(512, 512)
        self.struc_enc_512_512_1 = enc_block_v9(512, 512)
        self.struc_enc_inner_most_ = enc_block_v9(512, 512)

        self.struc_dec_inner_most_ = dec_block_v9(512, 512, use_dropout=False, pixel_shuffle=False)
        self.struc_dec_1024_512_1_ = dec_block_v9(1024, 512, use_dropout=False, pixel_shuffle=False, round = True)
        self.struc_dec_1024_512_2_ = dec_block_v9(1024, 512, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_1024_256_ = dec_block_v9(1024, 256, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_512_128_ = dec_block_v9(512, 128, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_256_128_ = dec_block_v9(256, 64, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_256_64_ = dec_block_v9(128, 64, use_dropout=False, pixel_shuffle=True)
        self.struc_dec_post_ = dec_post_block_v9(96, nc_output, pixel_shuffle=True, act_post=False)

    def forward(self, x):

        ## structure-to-image unet pass
        # endoer
        y0 = self.struc_enc_pre_(x)  # size: (n,32,192,192)
        y1 = self.struc_enc_32_64_(y0)  # size: (n,64,96,96)
        y2 = self.struc_enc_64_128_(y1)  # size: (n,128,48,48)
        y3 = self.struc_enc_128_256_(y2)  # size: (n,256,24,24)
        y4 = self.struc_enc_256_512(y3)  # size: (n,512,12,12)
        y5 = self.struc_enc_512_512_0(y4)  # size: (n,512,6,6)
        y6 = self.struc_enc_512_512_1(y5)  # size: (n,512,3,3)
        y7 = self.struc_enc_inner_most_(y6)  # size: (n,505,1,1)

        #concat emotion embedding
        # y7 = y7.squeeze(2) # size: (n,505,1)
        # y = torch.cat((y7, embedding.cuda()), dim = 1)# size: (n,512,1) embedding size: (n, 7 ,1)
        # y = y.unsqueeze(3) # size: (n,512,1,1)

        #decoder
        unet_dec_y = self.struc_dec_inner_most_(y7)  # size: (n,512,3,3)
        unet_dec_y = torch.cat([unet_dec_y, y6], 1)  # size: (n,1024,3,3)
        unet_dec_y = self.struc_dec_1024_512_1_(unet_dec_y)  # size: (n,512,6,6)
        unet_dec_y = torch.cat([unet_dec_y, y5], 1)  # size: (n,1024,6,6)
        unet_dec_y = self.struc_dec_1024_512_2_(unet_dec_y)  # size: (n,512,12,12)
        unet_dec_y = torch.cat([unet_dec_y, y4], 1)  # size: (n,1024,12,12)
        unet_dec_y = self.struc_dec_1024_256_(unet_dec_y)  # size: (n,256,24,24)
        unet_dec_y = torch.cat([unet_dec_y, y3], 1)  # size: (n,512,24,24)
        unet_dec_y = self.struc_dec_512_128_(unet_dec_y)  # size: (n,128,48,48)
        unet_dec_y = torch.cat([unet_dec_y, y2], 1)  # size: (n,256,48,48)
        unet_dec_y = self.struc_dec_256_128_(unet_dec_y)  # size: (n,128,96,96)
        unet_dec_y = torch.cat([unet_dec_y, y1], 1)  # size: (n,256,96,96)
        unet_dec_y = self.struc_dec_256_64_(unet_dec_y)  # size: (n,64,192,192)
        unet_dec_y = torch.cat([unet_dec_y, y0], 1)  # size: (n,128,192,192)
        unet_dec_y = self.struc_dec_post_(unet_dec_y)  # size: (n,3,384,384)
        output = unet_dec_y

        return output


class enc_pre_block_v9(nn.Module):
    def __init__(self, nc_input, nc_output):
        super(enc_pre_block_v9, self).__init__()
        self.model = []
        self.model += [weight_norm(nn.Conv2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True), name='weight', dim=None)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class enc_block_v9(nn.Module):
    def __init__(self, nc_input, nc_output):
        super(enc_block_v9, self).__init__()
        self.model = []
        self.model += [nn.LeakyReLU(0.2, True)]
        self.model += [weight_norm(nn.Conv2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True), name='weight', dim=None)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class dec_block_v9(nn.Module):
    def __init__(self, nc_input, nc_output, use_dropout=False, pixel_shuffle=False, round = False):
        super(dec_block_v9, self).__init__()
        self.model = []
        self.model += [nn.ReLU(True)]
        if pixel_shuffle:
            self.model += [weight_norm(nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True), name='weight', dim=None)]
            self.model += [nn.PixelShuffle(2)]
        elif round:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=5, stride=2, padding=1, bias=True)]
        if use_dropout:
            self.model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class dec_post_block_v9(nn.Module):
    def __init__(self, nc_input, nc_output, pixel_shuffle=False, act_post=True):
        super(dec_post_block_v9, self).__init__()
        self.model = []
        self.model += [nn.ReLU(True)]
        if pixel_shuffle:
            self.model += [weight_norm(nn.Conv2d(nc_input, 4*nc_output, kernel_size=3, stride=1, padding=1, bias=True), name='weight', dim=None)]
            self.model += [nn.PixelShuffle(2)]
        else:
            self.model += [nn.ConvTranspose2d(nc_input, nc_output, kernel_size=4, stride=2, padding=1, bias=True)]
        if act_post:
            self.model += [nn.Tanh()]
        else:
            self.model += [weight_norm(nn.Conv2d(nc_output, nc_output, kernel_size=3, stride=1, padding=1, bias=True), name='weight', dim=None)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class FaceDiscriminator(nn.Module):
    def __init__(self, params):
        super(FaceDiscriminator, self).__init__()
        self.model = [Conv2dBlock(3, 16, 1, 1, 0)]
        self.model += [DisBlock(16, 32)]
        self.model += [DisBlock(32, 64)]
        self.model += [DisBlock(64, 128)]
        self.model += [DisBlock(128, 256)]
        self.model += [DisBlock(256, 512)]
        self.model += [DisBlock(512, 1024)]
        self.model += [ToDisBlock(1024)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='in', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        # initialize normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DisBlock, self).__init__()
        model = [Conv2dBlock(input_dim, output_dim, 3, 1, 1, activation='lrelu')]
        # model += [Conv2dBlock(input_dim, output_dim, 3, 1, 1, activation='lrelu')]
        model += [DownSample(3, 2, 1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class ToDisBlock(nn.Module):
    def __init__(self, input_dim):
        super(ToDisBlock, self).__init__()
        self.conv1 = Conv2dBlock(1024, 1024, 3, 2, 1, activation='lrelu')
        self.conv2 = Conv2dBlock(1024, 1024, 3, 1, 0, activation='lrelu')
        self.fc = nn.Linear(1024, 1)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x


class DownSample(nn.Module):
    def __init__(self, kernel, stride, padding):
        super(DownSample, self).__init__()
        self.downsample = nn.AvgPool2d(kernel, stride, padding, count_include_pad=False)

    def forward(self, x):
        x = self.downsample(x)
        return x


'''Ganimation Generator'''
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim = 11, repeat_num=6):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(3):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(3):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.texture_reg = nn.Sequential(*layers)

        #layers = []
        #layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        #layers.append(nn.Sigmoid())
        #self.emotion_reg = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3)).cuda()
        x = torch.cat([x, c], dim=1)
        features = self.main(x)
        return self.texture_reg(features)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


'''Expression Transformer due to different emotion vector'''
class Expression_Transformer(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=7, repeat_num=3):
        super(Expression_Transformer, self).__init__()
        self._name = 'expression_transformer'

        layers = []
        layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.emotion_reg = nn.Sequential(*layers)

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3)).cuda()
        x = torch.cat([x, c], dim=1)
        features = self.main(x)
        return self.emotion_reg(features)
