import os
import sys
import pickle
import librosa
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

config = {
    'sample_interval': 1/30,
    'window_len': 0.025,
    'n_mfcc': 14,
    'fps': 30,
    'input_dir': './data/audio_train',
    'output_dir': './data/MFCC'
}


# get mfcc feature from audio file
def mfcc_from_file(audio_path, n_mfcc=config['n_mfcc'], sample_interval=config['sample_interval'], window_len=config['window_len']):
    # load audio to time serial wave, a 2D array
    audio, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=int(sample_interval*sr), n_fft=int(window_len*sr))
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs = np.concatenate((mfccs, mfccs_delta), axis=0)
    # MFCC mean and std
    mean, std = np.mean(mfccs, axis=0), np.std(mfccs, axis=0)
    mfccs = (mfccs-mean)/std
    padding_front = np.zeros((28, 15))
    padding_back = np.zeros((28, 15))
    front = np.column_stack((padding_front, mfccs))
    mfccs = np.column_stack((front, padding_back))
    return mfccs

def split_input_target(mfccs):
    #assert mfccs.shape[1] == parameters.shape[1], 'Squence length in mfcc and parameter is different'
    #assert phonemes.shape[1] == parameters.shape[1], 'Squence length in phoneme and parameter is different'
    seq_len = mfccs.shape[1]
    inputs = mfccs
    # target: parameter at a time, input: silding window (past 80, future 20)
    input_list = []
    for idx in range(15, seq_len-15):
        input_list.append(inputs[:, idx-15:idx+15])
    return input_list


if __name__ == '__main__':
    data_root = config['input_dir']
    audios = os.listdir(data_root)
    for audio in audios:
        audio_path = os.path.join(data_root, audio)# , sent['parameter']
        mfccs = mfcc_from_file(audio_path)
        input_list = split_input_target(mfccs)
        name_prefix = audio.replace('.wav', '') # maybe also ".mp3" and other files
        for idx in range(len(input_list)):
            input_data = input_list[idx]
            input_path = os.path.join(config['output_dir'], '%s_%03d.pickle'%(name_prefix, idx))

            with open(input_path, 'wb') as f:
                pickle.dump(input_data.T, f)



























