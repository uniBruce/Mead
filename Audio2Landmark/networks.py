import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import torch


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

###''' AT-B; AT-1; AT-2 '''

class ResNet_AT(nn.Module):
    def __init__(self, block, layers, end2end=True, at_type = 'AT-2'):
        self.inplanes = 64
        self.end2end = end2end
        super(ResNet_AT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)

        self.alpha = nn.Sequential(nn.Linear(512, 1),
                                   nn.Sigmoid())

        self.beta = nn.Sequential(nn.Linear(1024, 1),
                                  nn.Sigmoid())

        self.pred_fc1 = nn.Linear(512, 7)
        self.pred_fc2 = nn.Linear(1024, 7)
        self.at_type = at_type

        # self.threedmm_layer = threeDMM_model(alfa,threed_model_data)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x='', phrase='train', AT_level='second_level',vectors='',vm='',alphas_from1='',index_matrix=''):
        # print 'input image shape',x.shape
        vs = []
        alphas = []

        assert phrase == 'train' or phrase == 'eval'
        assert AT_level == 'first_level' or AT_level == 'second_level' or AT_level == 'pred'
        if phrase == 'train':
            num_pair = 3

            for i in range(num_pair):
                f = x[:, :, :, :, i]  # x[128,3,224,224]

                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)
                # f = self.dropout(f)
                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]

                # MN_MODEL(first Level)
                vs.append(f)
                alphas.append(self.alpha(self.dropout(f)))

            vs_stack = torch.stack(vs, dim=2)
            alphas_stack = torch.stack(alphas, dim=2)

            if self.at_type == 'AT-B':
                vm1 = vs_stack.sum(2).div(3)

            if self.at_type == 'AT-1':
                vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            if self.at_type == 'AT-2':
                vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))

                betas = []
                for i in range(len(vs)):
                    vs[i] = torch.cat([vs[i], vm1], dim=1)
                    betas.append(self.beta( self.dropout(vs[i])))
                cascadeVs_stack = torch.stack(vs, dim=2)
                betas_stack = torch.stack(betas, dim=2)
                # output = cascadeVs_stack.mul(betas_stack).sum(2).div(betas_stack.sum(2))
                ''' N2 '''
                self.type = 'N2'
                output = cascadeVs_stack.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))
                ''' rectify2 '''
                # self.type = 'rectify2'
                # output = vs_stack.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))
                ''' replace2 '''

            if self.at_type == 'AT-B' or self.at_type == 'AT-1':
                vm1 = self.dropout(vm1)
                pred_score = vm1

            if self.at_type == 'AT-2':
                output = self.dropout(output)
                output = output.unsqueeze(2).unsqueeze(3)
                pred_score =output

                # if output.shape[1] > 800 :  # 512 / 1024
                #     pred_score = self.pred_fc2(output)
                # else:
                #     pred_score = self.pred_fc1(output)

            return pred_score

        if phrase == 'eval':
            if AT_level == 'first_level':
                f = self.conv1(x)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)
                # f = self.dropout(f)
                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]
                # MN_MODEL(first Level)
                alphas = self.alpha(self.dropout(f))
                return f, alphas

            if AT_level == 'second_level':
                #print('Hello world')
                assert self.at_type == 'AT-2'

                vms = index_matrix.permute(1, 0).mm(vm)  # [381, 21783] -> [21783,381] * [381,512] --> [21783, 512]
                vs_cate = torch.cat([vectors, vms], dim=1)

                betas = self.beta(self.dropout(vs_cate))

                ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''


                ''' 2: beta *  '''
                # weight_catefc = vs_cate.mul(betas)  # [21570,512] * [21570,1] --->[21570,512]
                # sum_betas = index_matrix.mm(betas)  # [380,21570] * [21570,1] -> [380,1]
                # weightmean_catefc = index_matrix.mm(weight_catefc).div(sum_betas)
                ''' N2: alpha * beta '''
                # assert self.type == 'N2'
                print('HAHAHA')
                weight_catefc = vs_cate.mul(alphas_from1*betas)  # [21570,512] * [21570,1] --->[21570,512]
                alpha_beta = alphas_from1.mul(betas)
                sum_alphabetas = index_matrix.mm(alpha_beta)  # [380,21570] * [21570,1] -> [380,1]
                weightmean_catefc = index_matrix.mm(weight_catefc).div(sum_alphabetas)
                ''' rectify2 '''
                # assert self.type == 'rectify2'
                # weight_vectors = vectors.mul(alphas_from1*betas)
                # alpha_beta = alphas_from1.mul(betas)
                # sum_alphabetas = index_matrix.mm(alpha_beta)
                # weightmean_catefc = index_matrix.mm(weight_vectors).div(sum_alphabetas)
                ''' replace2 '''

                weightmean_catefc = self.dropout(weightmean_catefc)
                if weightmean_catefc.shape[1] > 800 :  # 512 / 1024
                    pred_score = self.pred_fc2(weightmean_catefc)
                else:
                    pred_score = self.pred_fc1(weightmean_catefc)

                return pred_score, betas

            if AT_level == 'pred':
                if self.at_type == 'AT-B' or self.at_type == 'AT-1':
                    pred_score = self.pred_fc1(self.dropout(vm))

                    return pred_score

                if self.at_type == 'AT-2':
                    print(" please use the setting of model( phrase='eval', AT_level='second_level') have achieved this design")


''' AT-B; AT-1; AT-2 '''
def resnet18_AT(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_AT(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)


    def forward(self, waveform):
        x = self.conv1(waveform)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)


        return x


class FusionGL(nn.Module):# input [B, 1, 3, 1024]
    def __init__(self):
        super(FusionGL, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv4 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv5 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv6 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(0,1), stride=(1,2))
        self.fc = nn.Linear(16, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        N, C, H, W= x.size()
        x = x.view(N, -1)
        x = self.fc(x)
        return x

class FusionAV(nn.Module):# input [B, 1, 6, 1024]
    def __init__(self):
        super(FusionAV, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(1,1), stride=(1,2))
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(0,1), stride=(1,2))
        self.conv4 = nn.Conv2d(1, 1, kernel_size=(3,4), padding=(0,1), stride=(1,2))
        self.conv5 = nn.Conv2d(1, 1, kernel_size=(2,4), padding=(0,1), stride=(1,2))
        self.conv6 = nn.Conv2d(1, 1, kernel_size=(1,4), padding=(0,1), stride=(1,2))
        self.fc = nn.Linear(16, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        N, C, H, W = x.size()
        x = x.view(N, -1)
        x = self.fc(x)
        return x


class EXCGEN(nn.Module):
    def __init__(self):
        super(EXCGEN, self).__init__()
        kernel_size = (7,3)
        stride = (2,1)
        padding = (1,1)
        #[20, 1, 1, 7] to [20, 1, 29 ,7] channel is always 1
        self.model = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size, stride, padding),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.ConvTranspose2d(1, 1, kernel_size, stride, padding),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.ConvTranspose2d(1, 1, kernel_size, stride, padding),
            nn.Tanh())

    def forward(self, x):
        x = self.model(x)
        return x


class EXCINT(nn.Module):
    def __init__(self,  param):
        super(EXCINT, self).__init__()
        in_channel = param['in_channel']
        mid_channel = param['mid_channel']
        out_channel = param['out_channel']
        self.fc1 = nn.Linear(in_channel, mid_channel)
        self.fc2 = nn.Linear(mid_channel, out_channel)

    def forward(self, x):
        N, C, H= x.size()
        x = x.view(N, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class EMCINT(nn.Module):
    def __init__(self,  param):
        super(EMCINT, self).__init__()
        in_channel = param['in_channel']
        mid_channel = param['mid_channel']
        out_channel = param['out_channel']
        self.fc1 = nn.Linear(in_channel, mid_channel)
        self.fc2 = nn.Linear(mid_channel, out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x


class Audio2Exp(nn.Module):
    def __init__(self, params):
        # initialize and input & output dimension
        super(Audio2Exp, self).__init__()
        self.input_dim = params['audio_dim']
        self.output_dim = params['pca_dim']
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

# class Flatten(nn.Module):
#     def __init__(self):
#         pass
#     def forward(self, x):
#         N, C, H, W = x.size() # read in N, C, H
#         return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image



