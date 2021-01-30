from utils_parallel import prepare_sub_folder, write_loss, write_log, get_config, Timer, write_image
from data import get_data_loader_list
import argparse
from torch.autograd import Variable
from trainer import GanimationTrainer
import torch.backends.cudnn as cudnn
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICE"]="0,1,2,3"

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_baseline_parallel.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

trainer = GanimationTrainer(config)
trainer.cuda()

train_loader = get_data_loader_list(config, train=True)
model_name = config['trainer']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

a = time.time()
iterations = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0
while True:
    for id, data in enumerate(train_loader):
        trainer.update_learning_rate()
        emc_label = data['EL'].cuda().detach()
        int_label = data['IL'].cuda().detach()
        ref_emc_label = data['REL'].cuda().detach()
        ref_int_label = data['RIL'].cuda().detach()
        video = data['V'].cuda().detach()
        video_ref = data['VR'].cuda().detach()


        # Main training code
        loss_gen, loss_content, loss_emc, loss_int, loss_recon, n2e_img, e2n_img = trainer.gan_update(video, emc_label, int_label, video_ref, ref_emc_label, ref_int_label, config)
        loss_dis = trainer.dis_update(n2e_img, video, e2n_img, video_ref, config)
        torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            log = "Iteration: %08d/%08d, Loss_gen: %f, Loss_content: %f, Loss_emc: %f, Loss_int: %f, Loss_recon: %f, Loss_tv: %f, Loss_dis: %f" % (iterations + 1, max_iter, loss_gen, loss_content, loss_emc, loss_int, loss_recon, loss_tv, loss_dis)
            print(log)
            write_log(log, output_directory)
            write_loss(iterations, trainer, train_writer)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        # Save generated images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                image_output = (video_ref, n2e_img, video)
                write_image(image_output, config['image_display_num'], image_directory, "%08d"%(iterations+1))


        iterations += 1
        if (iterations % 20) == 0:
            b=time.time() - a
            print('Iteration: %d , time spent: %f s'% (iterations,b))
        if iterations >= max_iter:
            sys.exit('Finish training')

