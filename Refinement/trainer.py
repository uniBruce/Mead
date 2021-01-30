import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
from utils_parallel import OneHot_emc_label, OneHot_int_label, weights_init, get_model_list, get_scheduler, vgg_preprocess, load_vgg16, load_unetenc, load_unetint, load_gan, mouth_center, mouth_extract, mouth_crop
from networks import Unet_V9_384x384, FaceDiscriminator


class GanimationTrainer(nn.Module):
    def __init__(self, param):
        super(GanimationTrainer, self).__init__()
        lr = param['lr']
        self.unetenc = load_unetenc(param['unetenc_path']) # Emotion class classifier
        self.unetint = load_unetint(param['unetenc_path']) # Emotion intensity classifier
        self.gan = load_gan(param['gan_path'])  # Ganimation generator
        self.dis = FaceDiscriminator(param).apply(weights_init(param['init']))
        self.encdec = Unet_V9_384x384().apply(weights_init(param['init'])) # Refinement Network
        self.vgg = load_vgg16(param['vgg_path']) # VGG Network
        for params in self.vgg.parameters():
            params.requires_grad = False

        self.gan_type = param['gan_type']
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        '''Refinement parameters'''
        model_gan_params = list(self.encdec.parameters())

        self.model_gan_opt = torch.optim.Adam([p for p in model_gan_params if p.requires_grad], lr=lr,
                                              betas=(param['beta1'], param['beta2']), weight_decay=param['weight_decay'])
        self.model_gan_scheduler = get_scheduler(self.model_gan_opt, param)

        '''Discriminator parameters'''
        model_dis_params = list(self.dis.parameters())

        self.model_dis_opt = torch.optim.Adam([p for p in model_dis_params if p.requires_grad], lr=lr,
                                        betas=(param['beta1'], param['beta2']), weight_decay=param['weight_decay'])
        self.model_dis_scheduler = get_scheduler(self.model_dis_opt, param)

    def update_learning_rate(self):
        if self.model_gan_scheduler is not None:
            self.model_gan_scheduler.step()
        if self.model_dis_scheduler is not None:
            self.model_dis_scheduler.step()

    '''Crop mouth'''
    def mouth_gen(self, video, landmark):
        cx, cy = mouth_center(landmark)
        mouth = mouth_extract(video, cx, cy, 128, 64)
        return mouth

    # def mouth_cropped(self, video, landmark):
    #     cx, cy = mouth_center(landmark)
    #     face = mouth_crop(video, cx, cy, 160, 100)
    #     return face

    '''Losses Part'''
    def criterion_emc(self,input, target):
        loss = nn.CrossEntropyLoss().cuda()
        return loss(input, target)

    def criterion_int(self,input, target):
        loss = nn.CrossEntropyLoss().cuda()
        return loss(input, target)

    def criterion_recon(self, input, target):
        loss = nn.L1Loss().cuda()
        return loss(input, target)

    def compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def vgg_loss(self, vgg, img, target, param):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        loss = 0
        for i in range(len(img_fea)):
            loss += param['vgg_w'][i] * torch.mean(torch.abs(img_fea[i]-target_fea[i]))
        return loss

    def calc_dis_loss(self, input_fake, input_real):
        outs0 = self.dis(input_fake)
        outs1 = self.dis(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(out0.data).cuda().detach()
                all1 = torch.ones_like(out1.data).cuda().detach()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.dis(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out0.data).cuda().detach()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


    def forward(self, video_ref, video_heatmap):
        self.eval()
        onehot = torch.tensor([[0., 0., 0., 0., 1., 0., 0.]]).unsqueeze(dim=2)
        cat_image = torch.cat((video_ref, video_heatmap), dim=1)
        talk_face = self.encdec(cat_image, onehot)

        return talk_face

    def gan_update(self, video, emc_label, int_label, video_heatmap, video_ref, mouth_ldmk, param):

        self.model_gan_opt.zero_grad()

        '''generate emotion image: forward process'''
        onehot_E_emc = OneHot_emc_label(emc_label)
        onehot_E_int = OneHot_int_label(int_label)
        onehot_E = torch.cat([onehot_E_emc, onehot_E_int], dim=1)

        '''neutral-to-emotion generation'''
        N2E_image = self.gan(video_ref, onehot_E)
        cat_image_N2E = torch.cat([N2E_image, video_heatmap], dim=1)
        N2E_tkfc = self.encdec(cat_image_N2E)
        mouth = self.mouth_gen(N2E_tkfc, mouth_ldmk)
        mouth_gt = self.mouth_gen(video, mouth_ldmk)

        '''loss part'''
        # content loss
        content_loss =  self.vgg_loss(self.vgg, N2E_tkfc, video,  param)

        # reconstruction loss
        recon_loss = self.criterion_recon(mouth, mouth_gt)

        # emotion classification loss
        emc_score_f = self.unetenc(N2E_tkfc)
        int_score_f = self.unetint(N2E_tkfc)

        emc_loss = self.criterion_emc(emc_score_f, emc_label)
        int_loss = self.criterion_int(int_score_f, int_label)

        # generator loss
        gen_loss = self.calc_gen_loss(N2E_tkfc)

        # total variation loss
        tv_loss = self.compute_loss_smooth(N2E_tkfc)

        gan_loss = gen_loss + content_loss + emc_loss + int_loss + param['rec_w']*recon_loss + param['tv_w']*tv_loss

        gan_loss.backward()
        self.model_gan_opt.step()

        return gen_loss, content_loss, emc_loss, int_loss, recon_loss, tv_loss, N2E_image, N2E_tkfc

    def dis_update(self, input_fake1, input_real1, param):
        self.model_dis_opt.zero_grad()
        # D loss
        self.loss_dis1 = param['gan_w'] * self.calc_dis_loss(input_fake1.detach(), input_real1)
        self.loss_dis = self.loss_dis1
        self.loss_dis.backward()
        self.model_dis_opt.step()
        return self.loss_dis

    def resume(self, checkpoint_dir, param):
        # Load Generator
        last_model_name = get_model_list(checkpoint_dir, "Gan")
        state_dict = torch.load(last_model_name)
        self.encdec.load_state_dict(state_dict['gan'])

        # Load Discriminator
        last_model_name = get_model_list(checkpoint_dir, "Dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])

        iterations = int(last_model_name[-11:-3])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_gan.pt'))
        self.model_gan_opt.load_state_dict(state_dict['gan_opt'])
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer_dis.pt'))
        self.model_dis_opt.load_state_dict(state_dict['dis_opt'])

        # Reinitilize schedulers
        self.model_gan_scheduler = get_scheduler(self.model_gan_opt, param, iterations)
        self.model_dis_scheduler = get_scheduler(self.model_dis_opt, param, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        '''Save generators, discriminators, and optimizers.'''

        model_gan_name = os.path.join(snapshot_dir, 'Gan_%08d.pt' % (iterations + 1))
        model_dis_name = os.path.join(snapshot_dir, 'Dis_%08d.pt' % (iterations + 1))

        opt_gan_name = os.path.join(snapshot_dir, 'optimizer_gan.pt')
        opt_dis_name = os.path.join(snapshot_dir, 'optimizer_dis.pt')

        torch.save({'gan':self.encdec.state_dict()}, model_gan_name)
        torch.save({'dis': self.dis.state_dict()}, model_dis_name)

        torch.save({'gan_opt': self.model_gan_opt.state_dict()}, opt_gan_name)
        torch.save({'dis_opt': self.model_dis_opt.state_dict()}, opt_dis_name)




