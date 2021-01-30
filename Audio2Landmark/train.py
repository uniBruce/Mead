from utils import prepare_sub_folder, write_loss, write_log, get_config, Timer
from data import get_data_loader_list
import argparse
from torch.autograd import Variable
from trainer import LipTrainer
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
parser.add_argument('--config', type=str, default='configs.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
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

train_loader = get_data_loader_list(config, train=True)
model_name = config['trainer']
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

iterations = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0
while True:
    for id, data in enumerate(train_loader):
        trainer.update_learning_rate()
        audio = data['AU'].cuda(async=True).detach()
        parameter = data['PM'].cuda(async=True).detach()

        # Main training code
        loss_pca = trainer.trainer_update(audio, parameter)
        torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            log = "Iteration: %08d/%08d, Loss_exc: %f" % (iterations + 1, max_iter, loss_pca)
            print(log)
            write_log(log, output_directory)
            write_loss(iterations, trainer, train_writer)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
