from utils import prepare_sub_folder, write_loss, write_log, get_config, Timer, draw_heatmap_from_78_landmark, save_image
from data1 import get_data_loader_list
import argparse
from torch.autograd import Variable
from trainer1 import LipTrainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs_test.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--checkpoint_lstm', type=str, default='./outputs/LipTrainer/checkpoints/audio2exp_00200000.pt', help='Path to the checkpoint file')
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']


if config['trainer'] == 'LipTrainer':
    trainer = LipTrainer(config)
else:
    sys.exit('Train option not supported')
trainer.cuda()

state_dict_lstm = torch.load(opts.checkpoint_lstm)
trainer.audio2exp.load_state_dict(state_dict_lstm['audio2exp'])

train_loader = get_data_loader_list(config, train=False)
model_name = config['trainer']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))
f =open(config['pca_path'], 'rb')
pca = pickle.load(f)

iterations = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0
while True:
    for id, data in enumerate(train_loader):
        trainer.update_learning_rate()
        audio = data['AU'].cuda(async=True).detach()
        ldmk = data['PM']

        # Main testing code
        with torch.no_grad():
            pca_ldmk = trainer.forward(audio, ldmk).cpu()
            torch.cuda.synchronize()
            fake_ldmk = pca.inverse_transform(pca_ldmk)[0]
            #print(ldmk)
            fake_heatmap = draw_heatmap_from_78_landmark(fake_ldmk, 384, 384)
            real_heatmap = draw_heatmap_from_78_landmark(ldmk, 384, 384)

            image_output = [fake_heatmap, real_heatmap]
            save_image(image_output, image_directory, "%06d" % (iterations + 1))

            iterations += 1
            if iterations >= len(train_loader):
                sys.exit('Finish testing')
