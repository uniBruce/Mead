import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path
from utils import OneHot, weights_init, get_model_list, get_scheduler, Dict_Unite
from networks1 import resnet18_AT, SoundNet, FusionAV, FusionGL, EXCGEN, EXCINT, Audio2Exp, EMCINT



class LipTrainer(nn.Module):
    def __init__(self, param):
        super(LipTrainer, self).__init__()
        lr = param['lr']
        # Initiate the networks

        self.audio2exp = Audio2Exp(param)

        a2e_params = list(self.audio2exp.parameters())

        self.a2e_opt = torch.optim.Adam([p for p in a2e_params if p.requires_grad], lr=lr,
                                        betas=(param['beta1'], param['beta2']), weight_decay=param['weight_decay'])

        self.a2e_scheduler = get_scheduler(self.a2e_opt, param)

        # Network weight initialization
        self.apply(weights_init(param['init']))


    def criterion_pca(self,input, target):
        return torch.mean(F.pairwise_distance(input.double(), target.double())).cuda()

    def forward(self, audio):
        self.eval()
        exp_pred = self.audio2exp(audio)
        return exp_pred

    def trainer_update(self, audio, parameter):
        self.a2e_opt.zero_grad()

        pca_pred = self.audio2exp(audio)
        self.loss_exc = self.criterion_pca(pca_pred, parameter)
        self.loss_exc.backward()
        self.a2e_opt.step()

        return self.loss_exc

    def update_learning_rate(self):
        if self.a2e_scheduler is not None:
            self.a2e_scheduler.step()


    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, 'audio2exp')
        state_dict = torch.load(last_model_name)
        self.audio2exp.load_state_dict(state_dict['audio2exp'])
        iterations = int(last_model_name[-11:-3])

        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.a2e_opt.load_state_dict(state_dict['a2e_opt'])

        # Reinitilize schedulers
        self.a2e_scheduler = get_scheduler(self.a2e_opt, param, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        a2e_name = os.path.join(snapshot_dir, 'audio2exp_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'audio2exp': self.audio2exp.state_dict()}, a2e_name)
        torch.save({'a2e_opt': self.a2e_opt.state_dict()}, opt_name)
