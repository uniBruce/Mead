import torchfile
from torch.optim import lr_scheduler
from networks import Vgg16, Unet_Enc_384x384, Unet_Int_384x384
import torchvision.utils as vutils
import torch
import os
import numpy as np
import math
import yaml
import torch.nn.init as init
import time
import librosa
import cv2
from torchvision import transforms
import torch.distributed
from torch.nn.parameter import Parameter

# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# load_inception
# vgg_preprocess
# get_scheduler
# weights_init

def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    num_gpus = torch.cuda.device_count()
    #torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()


def get_ip(ip_str):
    """
    input format: SH-IDC1-10-5-30-[137,152] or SH-IDC1-10-5-30-[137-142,152] or SH-IDC1-10-5-30-[152, 137-142]
    output format 10.5.30.137
    """
    import re
    # return ".".join(ip_str.replace("[", "").split(',')[0].split("-")[2:])
    return ".".join(re.findall(r'\d+', ip_str)[1:5])


# write image to visualize
def write_image(image_outputs, display_image_num, image_directory, postfix):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    #image_outputs = [images for images in image_outputs]
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, filename='%s/gen_%s.jpg'%(image_directory, postfix), nrow=1)

def save_image(image_outputs, image_name, image_directory):
    image_path = os.path.join(image_directory, image_name)
    cv2.imwrite(image_path, image_outputs)

# tensor to cv2 image
def tensor_to_cv2(tensor, image_name, image_directory):
    tensor = torch.clamp(tensor, min=-1, max=1)
    transform_list = [transforms.Normalize((-1, -1, -1), (2, 2, 2)), transforms.ToPILImage()]
    transform = transforms.Compose(transform_list)
    img = transform(tensor)
    # img.save('fake_pil.jpg')
    # Convert RGB to BGR
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    image_path = os.path.join(image_directory, image_name)
    print(image_path)
    cv2.imwrite(image_path, img)


# pad mfcc array
def pad_mfcc(mfcc, append=False, target_shape=(100,28)):
    mfcc_pad = np.zeros(target_shape, dtype=np.float32)
    if append:
        mfcc_pad[:mfcc.shape[0],:] = mfcc
    else:
        mfcc_pad[-mfcc.shape[0]:,:] = mfcc
    return mfcc_pad


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


# def OneHot(pred_score, exc_map):
#     one_hot = torch.zeros((pred_score.shape[0], pred_score.shape[1]))
#     for i in range(pred_score.shape[0]):
#         peak, index = torch.max(pred_score[i], 0)
#         one_hot[i][index] = 1
#     for i in range(exc_map.shape[0]):
#         # one_map = exc_map[i]
#         for j in range(exc_map[i].shape[1]):
#             a = 2 * one_hot[i][j] - 1
#             exc_map[i][:, j] = exc_map[i][:, j] * a
#     return exc_map

def OneHot(score):
    onehot_vector = torch.zeros((score.shape[0], score.shape[1]))
    emotion_indices = torch.argmax(score, dim=1)
    onehot_vector[torch.tensor(range(score.shape[0]), dtype=torch.int64), emotion_indices] = 1
    return onehot_vector

def OneHot_emc_label(label):
    onehot_vector = torch.zeros((label.shape[0], 8))
    onehot_vector[torch.tensor(range(label.shape[0]), dtype=torch.int64), label] = 1
    return onehot_vector

def OneHot_int_label(label):
    onehot_vector = torch.zeros((label.shape[0], 3))
    onehot_vector[torch.tensor(range(label.shape[0]), dtype=torch.int64), label] = 1
    return onehot_vector

def dict_unite(pretrained_state_dict, model_state_dict):
    for key in pretrained_state_dict:
        if 'module.' in key:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
    return model_state_dict

def write_log(log, output_directory):
    with open(os.path.join(output_directory, 'log.txt'), 'a') as f:
        f.write(log+'\n')


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


# draw mouth heatmap from landmark coordinates
def draw_heatmap_from_78_landmark(batchsize, landmarks, width, height):
    heat_maps = []
    # draw lines
    def draw_line(start_idx, end_idx, landmark, heatmap):
        for pts_idx in range(start_idx, end_idx):
            cv2.line(heatmap, (int(landmark[pts_idx * 2]), int(landmark[pts_idx * 2 + 1])),
                     (int(landmark[pts_idx * 2 + 2]), int(landmark[pts_idx * 2 + 3])), thickness=3, color=255)
    for i in range(batchsize):
        heatmap = np.zeros((width, height), dtype=np.uint8)
        landmark = landmarks[i,:]
        draw_line(84-84+19, 90-84+19, landmark, heatmap)     # upper outer
        draw_line(96-84+19, 100-84+19, landmark, heatmap)   # upper inner
        draw_line(100-84+19, 103-84+19, landmark, heatmap)   # lower inner
        draw_line(90-84+19, 95-84+19, landmark, heatmap)    # lower outer
        draw_line(0, 18, landmark, heatmap)    # jaw line
        cv2.line(heatmap, (int(landmark[(96-84+19) * 2]), int(landmark[(96-84+19) * 2 + 1])),
                 (int(landmark[(103-84+19) * 2]), int(landmark[(103-84+19) * 2 + 1])), thickness=3, color=255)
        cv2.line(heatmap, (int(landmark[(84-84+19) * 2]), int(landmark[(84-84+19) * 2 + 1])),
                 (int(landmark[(95-84+19) * 2]), int(landmark[(95-84+19) * 2 + 1])), thickness=3, color=255)
        heatmap = cv2.GaussianBlur(heatmap, ksize=(5, 5), sigmaX=1, sigmaY=1)
        heatmap = torch.FloatTensor(heatmap)
        heatmap = heatmap.unsqueeze(0).unsqueeze(1)
        heat_maps.append(heatmap)

    map = heat_maps[0]
    for i in range(1, batchsize):
        map = torch.cat((map, heat_maps[i]), dim = 0)

    return map


def mouth_center(mouth_landmark):
    center_x = int((mouth_landmark[:, 0] + mouth_landmark[:, 12]) / 2)
    center_y = int((mouth_landmark[:, 1] + mouth_landmark[:, 13]) / 2)
    return center_x, center_y

# paste a white mask image on base image
# def mask_image(base_img, center_x, center_y, mask_width, mask_height):
#     start_x = center_x - int(mask_width/2)
#     start_y = center_y - int(mask_height/2)
#     masked_img = base_img.copy()
#     mask = 255 * np.ones((mask_height, mask_width, 3), dtype=np.uint8)
#     masked_img[start_y:start_y+mask_height, start_x:start_x+mask_width, :] = mask
#     return masked_img


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch - mean # subtract mean
    return batch


def load_vgg16(model_path):
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(model_path))
    return vgg

def load_unetenc(model_path):
    unetenc = Unet_Enc_384x384()
    model = torch.load(model_path)['unetenc']
    #pretrained_state_dict = torch.load(model_path)['unetenc']
    #model_state_dict = unetenc.state_dict()
    #model = dict_unite(pretrained_state_dict, model_state_dict)
    unetenc.load_state_dict(model)
    return unetenc

def load_unetint(model_path):
    unetint = Unet_Int_384x384()
    model = torch.load(model_path)['unetint']
    #pretrained_state_dict = torch.load(model_path)['unetint']
    #model_state_dict = unetenc.state_dict()
    #model = dict_unite(pretrained_state_dict, model_state_dict)
    unetint.load_state_dict(model)
    return unetint
