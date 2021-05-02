import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os.path
import numpy as np
import random
import librosa
import pickle
from PIL import Image
import torchvision.transforms as transforms


def path_extractor(name): #JK_a01_000.jpeg
    name = name.split('.')[0]
    character = name.split('_')[0]
    folder = name.split('_')[1]
    path = character + '/' + folder +'/'
    return path

def default_image_loader(path, gray = False):
    if gray:
        return Image.open(path).convert('L')
    else:
        return Image.open(path).convert('RGB')

def default_audioG_loader(path):
    fs = 44100
    y = librosa.load(path, fs)[0]
    fn = int(fs ** 2 / y.shape[0])
    y_down = librosa.resample(y, fs, fn)
    delta = fs - y_down.shape[0]
    for i in range(delta):
        y_down = np.append(y_down, y_down[-1*i])
    return y_down


def default_audioL_loader(path):
    fs = 44100
    return librosa.load(path, fs)[0]

def default_pickle_loader(path):
    with open(path, 'rb') as rf:
        data = pickle.load(rf)
    return data


def default_audiolist_reader(flist):
    sample_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            audio_gpath = line.strip().split(' ')[0]
            audio_local_sample = line.strip().split(' ')[1:]
            audio_local_path = path_extractor(audio_local_sample[0])
            audio_local_path_0 = audio_local_path + audio_local_sample[0]
            audio_local_path_1 = audio_local_path + audio_local_sample[1]
            audio_local_path_2 = audio_local_path + audio_local_sample[2]
            sample_list.append([audio_gpath, audio_local_path_0, audio_local_path_1, audio_local_path_2])

    return sample_list


def default_imagelist_reader(flist):
    sample_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            image_path = line.strip().split(' ')[0]
            emc_label = line.strip().split(' ')[1]
            int_label = line.strip().split(' ')[2]
            ne_image_path = line.strip().split(' ')[3]
            ref_emc_label = line.strip().split(' ')[4]
            ref_int_label = line.strip().split(' ')[5]

            sample_list.append([image_path, emc_label, int_label, ne_image_path, ref_emc_label, ref_int_label])

    return sample_list

def make_dataset(dir):
    data_paths = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        data_paths.append(os.path.join(dir, fname))
    return data_paths

def default_picklelist_reader(flist):
    pickle_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            pickle = line.strip().split()
            pickle_list.append(pickle)
    return pickle_list

def default_parameter_reader(flist):
    parameter_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            parameters = line.split(' ')[1:]
            for i in range(len(parameters)):
                parameters[i] = float(parameters[i])
            parameter_list.append(parameters)
    return parameter_list

def default_mouth_ldmk_reader(flist):
    parameter_list = []
    name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            name = line.strip().split()[-1]
            name_list.append(name)
            parameters = line.strip().split()[:-1]
            parameter_mouth = parameters[168:182]
            parameter_list.append(parameter_mouth)
    return name_list, parameter_list

def default_comparameter_reader(flist):
    parameter_list = []
    name_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            name = line.strip().split()[0]
            name_list.append(name)
            parameters = line.strip().split()[1:]
            parameter_downpart = parameters[14:52] + parameters[168:208]
            parameter_list.append(parameter_downpart)
    return name_list, parameter_list

def default_alphalist_reader(flist):
    alpha_list = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            alpha = float(line.strip())
            alpha_list.append(alpha)
    return alpha_list

def img_concat_tool(data1, data2, data3):
    input = torch.zeros([data1.shape[0], data1.shape[1], data1.shape[2], 3])
    input[:, :, :, 0] = data1
    input[:, :, :, 1] = data2
    input[:, :, :, 2] = data3
    return input

def aud_concat_tool(data1, data2, data3):
    input = np.append(data1, data2)
    input = np.append(input, data3)
    return input

def preprocess_tool(raw_audio, config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    # Make range [-256, 256]
    peak = abs(max(abs(np.max(raw_audio)), abs(np.min(raw_audio))))
    raw_audio = raw_audio/peak
    raw_audio *= 256.0

    # Make minimum length available
    length = config
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(length/raw_audio.shape[0] + 1))
        if length < raw_audio.shape[0]:
            raw_audio = raw_audio[:length]

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, -1, 1])

    return raw_audio.copy()

class CaffeCrop(object):
    """
    This class take the same behavior as sensenet
    """
    def __init__(self, phase, final_size):
        assert (phase == 'train' or phase == 'test')
        self.phase = phase
        self.final_size = final_size

    def __call__(self, img):
        # pre determined parameters
        final_size = self.final_size
        final_width = final_height = final_size
        res_img = img.resize((final_width, final_height))
        return res_img


class MeadDataset(data.Dataset):
    def __init__(self, root, flist, transform=None, transform_gray = None, audio_transform = None, preprocess=True, image_loader = default_image_loader, config = None, pickle_loader =default_pickle_loader,
                 imagelist_reader = default_imagelist_reader, landmarklist_reader = default_comparameter_reader, mouth_ldmk_list_reader = default_mouth_ldmk_reader, preprocess_tool = preprocess_tool):
        self.root = root
        self.video_root = os.path.join(self.root, 'Video')
        self.audio_root = os.path.join(self.root, 'Audio')
        self.heatmap_root = os.path.join(self.root, 'Heatmap')

        self.video_list = flist['video_list']
        self.landmark_list = flist['landmark_list']
        self.phoneme_list = flist['phoneme_list']

        self.transform = transform
        self.transform_gray = transform_gray
        self.audio_transform = audio_transform
        self.preprocess = preprocess
        self.config = config
        self.image_loader = image_loader
        self.pickle_loader = pickle_loader
        self.imagelist_reader = imagelist_reader
        self.landmarklist_reader = landmarklist_reader
        self.mouth_ldmk_list_reader = mouth_ldmk_list_reader
        self.preprocess_tool = preprocess_tool
        self.video_data = self.imagelist_reader(self.video_list)
        self.image_name, self.landmark_data = self.mouth_ldmk_list_reader(self.landmark_list)

    def __getitem__(self, index):
        video_sample = self.video_data[index]
        video_path = os.path.join(self.video_root, video_sample[0])
        emc_label = int(video_sample[1])
        int_label = int(video_sample[2])
        video_ref_path = os.path.join(self.video_root, video_sample[3])
        video_heatmap_path = os.path.join(self.heatmap_root, video_sample[0])

        video = self.image_loader(video_path)
        video_ref = self.image_loader(video_ref_path)
        video_heatmap = self.image_loader(video_heatmap_path, True)

        if self.transform is not None and self.transform_gray is not None:
            video = self.transform(video)
            video_ref = self.transform(video_ref)
            video_heatmap = self.transform_gray(video_heatmap)

        landmark = self.landmark_data[index]
        landmark = np.array(landmark, dtype=float)


        return {'V':video, 'EL':emc_label, 'IL':int_label, 'VR':video_ref, 'VH': video_heatmap, 'LM':landmark}

    def __len__(self):
        return len(self.video_data)


class MeadTestDataset(data.Dataset):
    def __init__(self, root, flist, transform=None, audio_transform = None, preprocess=True, image_loader = default_image_loader, config = None, pickle_loader =default_pickle_loader,
                 imagelist_reader = default_imagelist_reader, landmarklist_reader = default_comparameter_reader, preprocess_tool = preprocess_tool):
        self.root = root
        self.video_root = os.path.join(self.root, 'Video')
        self.audio_root = os.path.join(self.root, 'Audio')
        self.heatmap_root = os.path.join(self.root, 'Heatmap')

        self.video_list = flist['video_list_test']

        self.transform = transform
        self.audio_transform = audio_transform
        self.preprocess = preprocess
        self.config = config
        self.image_loader = image_loader
        self.pickle_loader = pickle_loader
        self.imagelist_reader = imagelist_reader
        self.landmarklist_reader = landmarklist_reader
        self.preprocess_tool = preprocess_tool
        self.video_data = self.imagelist_reader(self.video_list)

    def __getitem__(self, index):
        video_sample = self.video_data[index]

        video_ref_path = os.path.join(self.video_root, video_sample[2])
        video_heatmap_path = os.path.join(self.heatmap_root, video_sample[0])
        video_name = video_sample[0]

        video_ref = self.image_loader(video_ref_path)
        video_heatmap = self.image_loader(video_heatmap_path).convert('L')

        if self.transform is not None:
            video_ref = self.transform(video_ref)
            video_heatmap = self.transform(video_heatmap)

        return {'VR':video_ref, 'VH': video_heatmap, 'VN': video_name} #

    def __len__(self):
        return len(self.video_data)


class DemoTestDataset(data.Dataset):
    def __init__(self, root, flist, transform=None, transform_gray = None, audio_transform = None, preprocess=True, image_loader = default_image_loader, config = None, pickle_loader =default_pickle_loader,
                 landmarklist_reader = default_comparameter_reader, picklelist_reader = default_picklelist_reader, preprocess_tool = preprocess_tool):
        self.root = root
        self.video_root = os.path.join(self.root, 'Reference')
        self.audio_root = './MFCC_test'

        self.audio_list = flist['audio_list_test']

        self.transform = transform
        self.transform_gray = transform_gray
        self.audio_transform = audio_transform
        self.preprocess = preprocess
        self.config = config
        self.image_loader = image_loader
        self.pickle_loader = pickle_loader
        self.landmarklist_reader = landmarklist_reader
        self.picklelist_reader = picklelist_reader
        self.preprocess_tool = preprocess_tool
        self.audio_data = self.picklelist_reader(self.audio_list)

    def __getitem__(self, index):
        sample = self.audio_data[index]
        audio_path = os.path.join(self.audio_root, sample[0])
        file = sample[0].split('/')[0]
        em = file.split('_')[1] # here we use the folder number to represent emotion, please see trainer_demo.py for more details
        level = file.split('_')[2]
        audio = self.pickle_loader(audio_path)
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        video_ref_path = os.path.join(self.video_root, sample[1])
        video_ref = self.image_loader(video_ref_path)

        if self.transform is not None:
            video_ref = self.transform(video_ref)

        return {'VR':video_ref, 'AU':audio, 'EM':em, 'LV':level } # 'VN': video_name 'VH': video_heatmap

    def __len__(self):
        return len(self.audio_data)


def get_data_loader_list(config, train, demo):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
    transform_gray = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    transform_list = [transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))]
    audio_transform = transforms.Compose(transform_list)

    if train == True:
        dataset = MeadDataset(config['root'], config['flist'], transform=transform, transform_gray = transform_gray, audio_transform = audio_transform, config=config['audio_load_size'])
        #datasampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=train, drop_last=True, num_workers=config['num_workers'])
    else:
        if demo == False:
            dataset = MeadTestDataset(config['root'], config['flist'], transform=transform, audio_transform=audio_transform, config=config['audio_load_size'])
            loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=train, drop_last=True, num_workers=config['num_workers'])
        else:
            dataset = DemoTestDataset(config['root'], config['flist'], transform=transform, audio_transform=audio_transform, config=config['audio_load_size'])
            loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=train, drop_last=True, num_workers=config['num_workers'])

    return loader
