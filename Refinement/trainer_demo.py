import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
import pickle
import torchvision.transforms as transforms
from utils_parallel import OneHot_emc_label, OneHot_int_label, weights_init, get_model_list, get_scheduler, vgg_preprocess, load_gan, draw_heatmap_from_78_landmark
from networks import Unet_V9_384x384, FaceDiscriminator, Generator, Expression_Transformer, Unet_Int_384x384, Unet_Enc_384x384, Audio2Exp


class GanimationTrainer(nn.Module):
    def __init__(self, param):
        super(GanimationTrainer, self).__init__()
        self.audio2exp = Audio2Exp(param)
        self.dis = FaceDiscriminator(param).apply(weights_init(param['init']))
        self.gan = load_gan(param['gan_path']) # emotion intensity classifier
        self.encdec = Unet_V9_384x384().apply(weights_init(param['init']))

        self.gan_type = param['gan_type']
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        f = open(param['pca_path'], 'rb')
        self.pca = pickle.load(f)

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def page2emo(self, page):
        if 1 <= int(page) <= 8:
            emc = 0 # angry
        elif 9 <= int(page) <= 16:
            emc = 1 # disgust
        elif 17 <= int(page) <= 24:
            emc = 2 # contempt
        elif 25 <= int(page) <= 33:
            emc = 3 # fear
        elif 34 <= int(page) <= 43:
            emc = 4 # happy
        elif 44 <= int(page) <= 52:
            emc = 5 # sad
        elif 53 <= int(page) <= 61:
            emc = 6 # surprised
        else:
            emc = 7 # neutral
        return emc

    def forward(self, video_ref, audio, em, level):
        self.eval()
        onehot = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        em_index = self.page2emo(em)
        level_index = int(level)
        onehot[0][em_index] = 1.
        onehot[0][level_index+7] = 1.

        # Or you can just set up the emotion here
        #onehot = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]) -> emotion: angry intensity:3

        em_image = self.gan(video_ref, onehot)
        pca_ldmk = self.audio2exp(audio).cpu()
        fake_ldmk = self.pca.inverse_transform(pca_ldmk)[0]
        heatmap = self.transform(draw_heatmap_from_78_landmark(fake_ldmk, 384, 384))
        video_heatmap = torch.Tensor(heatmap).unsqueeze(0).cuda()
        cat_image = torch.cat((em_image, video_heatmap), dim=1)
        talk_face = self.encdec(cat_image)

        return talk_face




