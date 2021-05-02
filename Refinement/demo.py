from utils_parallel import prepare_sub_folder,  get_config, save_image, write_image, tensor_to_cv2, dict_unite
from data import get_data_loader_list
import argparse
from trainer_demo import GanimationTrainer
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

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_demo.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./demo', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--checkpoint_ref', type=str, default='./Gan_00171900.pt', help='Path to the refinement checkpoint file')
parser.add_argument('--checkpoint_n2e', type=str, default='./Gan_00203500.pt', help='Path to the neutral2emotion checkpoint file')
parser.add_argument('--checkpoint_lstm', type=str, default='./audio2exp_00200000.pt', help='Path to the audio2landmark checkpoint file')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

trainer = GanimationTrainer(config)
trainer.cuda()

# Load audio2landmark module
state_dict_lstm = torch.load(opts.checkpoint_lstm)
trainer.audio2exp.load_state_dict(state_dict_lstm['audio2exp'])
# Load neutral2emotion module
state_dict_gan = torch.load(opts.checkpoint_n2e)
trainer.gan.load_state_dict(state_dict_gan['gan'])
# Load refinement module
state_dict_gan = torch.load(opts.checkpoint_ref)
trainer.encdec.load_state_dict(state_dict_gan['gan'])

trainer.eval()

test_loader = get_data_loader_list(config, train=False, demo=True)
model_name = config['trainer']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with torch.no_grad():
    for id, data in enumerate(test_loader):
        video_ref = data['VR'].cuda().detach()
        audio = data['AU'].cuda().detach()
        em = data['EM'][0]
        level = data['LV'][0]

        # Main training code
        tkfc = trainer.forward(video_ref, audio, em, level)#[0].cpu()
        torch.cuda.synchronize()
        sys.stdout.write('\rTest data progress: %08d/%08d' % (id + 1, len(test_loader)))

        image_output = (tkfc)
        em_fc = trainer.page2emo(em)
        image_dir = os.path.join(image_directory, str(em_fc))
        if os.path.exists(image_dir):
            pass
        else:
            os.mkdir(image_dir)
        write_image(image_output, config['image_display_num'], image_dir, "%03d"%(id+1))

        id += 1
        if id >= len(test_loader):
            sys.exit('Finish testing')

