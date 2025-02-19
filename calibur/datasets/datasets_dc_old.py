# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from pathlib import Path
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from asteroid.data import LibriMix
import random
import torch
import soundfile as sf
import pyloudnorm

def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, 'r') as f:
        f.readline() # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(',')
            aux_it = iter(aux)
            aux = [(auxpath,int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[mix_id][utt_id] = aux
    return data

class LibriMixInformed_dc(Dataset):
    def __init__(
        self, csv_dir, utt_scp_file, spk_list, task="sep_clean", sample_rate=16000, n_src=2,
        segment=3, segment_aux=3, 
        ):
        self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, segment)
        self.seg_len = self.base_dataset.seg_len

        self.sample_rate = sample_rate
        self.spk_list, self.num_spk= self._load_spk(spk_list)
        # print("spk_list:",self.spk_list)

        self.chunk_size = int(segment * sample_rate)
        self.aux_chunk_size  = self.chunk_size
        self.comm_length = self.chunk_size

        self.seg_least= int(self.chunk_size // 2 )  
        self.EPS = 1e-10

        self.data , self.data_spk = self._load_data(utt_scp_file)
        # self.speeds = [0.9, 1.0, 1.1]
    
        self.total_lines = len(self.data)

        self.meter = pyloudnorm.Meter(self.sample_rate)
        self.MAX_AMP = 0.9
        # self.MIN_LOUDNESS = -33
        # self.MAX_LOUDNESS = -25


    def __len__(self):
        return len(self.data)

    def _get_segment_start_stop(self, seg_len, length, drop=0):
        if seg_len is not None:
            start = random.randint(0, length - seg_len - drop)
            stop = start + seg_len
        else:
            start = 0
            stop = None
        return start, stop

    def _load_data(self, scp_file, is_spk=True):
        with open(scp_file, 'r') as f:
            lines = f.readlines()
        data = []
        
        spk_hashtable = {}
        for line in lines:
            parts = line.strip().split()
            sentence_id = parts[0]
            spk_id = (sentence_id.split('_')[-1]).split('-')[0]
            sentence_path = parts[1]
            data_len = int(parts[2])
            if data_len < self.seg_least:
                continue
            if is_spk:
                spk_id = self.spk_list.index(spk_id)
                data.append((sentence_id, spk_id, sentence_path, data_len))
                if spk_id not in spk_hashtable.keys():
                    spk_hashtable[spk_id] = [(sentence_id, sentence_path, data_len)]
                else:
                    spk_hashtable[spk_id].append((sentence_id, sentence_path, data_len))
            else:
                data.append((sentence_id, sentence_path, data_len))
        return data, spk_hashtable


    def _load_wav(self, path, length, drop=0):
        if length < self.comm_length:
            wav,_ = sf.read(path, dtype="float32")
            # if len(wav.shape) != 1:
            #     wav = wav[:, np.random.randint(0, wav.shape[1])]
            padding_length = self.comm_length - length
            #  np.pad  0
            wav = np.pad(wav, (0, padding_length), 'constant')
        else:
        # start to random start
            start, stop = self._get_segment_start_stop(self.comm_length, length, drop)
            wav,_ = sf.read(path, dtype="float32", start=start, stop=stop)
            # wav, _ = librosa.load(path, offset=start, duration=self.comm_length)
        # if len(wav.shape) != 1:
        #     wav = wav[:, -1]  

        return wav

    def _generate_noise(self,spk_id,type):
        # just other spk
        other_spks = [spk for spk in self.data_spk.keys() if spk != spk_id]
        random_spk = np.random.choice(other_spks, 1)[0]
        element = random.choice(self.data_spk[random_spk])
        noise_id = element[0]
        path = element[1]
        length = element[2]
        noise = self._load_wav(path, length)   

        # if type == 'other':
        #noise_id, noise = self.get_no_silent_wav(self.data_spk[random_spk])

        # elif type == 'noise':
        #     noise_id, noise = self.get_no_silent_wav(self.data_noise_spk[random_spk])

        return noise, noise_id
    # def get_no_silent_wav(self, data):

    #     element = random.choice(data)
    #     wav_id = element[0]
    #     path = element[1]
    #     length = element[2]
    #     wav = self._load_wav(path, length)
        
    #     c_loudness = self.meter.integrated_loudness(wav)
    #     if c_loudness == float('-inf'):
    #         return self.get_no_silent_wav(data)
    #     # wav = self.normalize(wav)
    #     return wav_id, wav


    def _load_spk(self, spk_list_path):
        if spk_list_path is None:
            return []
        lines = open(spk_list_path).readlines()
        new_lines = []
        for line in lines:
            new_lines.append(line.strip())

        return new_lines, len(new_lines)

    
    def _mix(self, sources_list):

        mix_length = self.comm_length
        mixture = np.zeros(mix_length, dtype=np.float32)
        for i, _ in enumerate(sources_list):
            mixture += sources_list[i]
        return mixture


    def __getitem__(self, idx):

        spkwav_id, spk_id, spkwav_path, spkwav_length = self.data[idx]
        spkwav = self._load_wav(spkwav_path, spkwav_length)
        source = torch.from_numpy(spkwav)[None]

        available_enrolls = [enroll_data for enroll_data in self.data_spk[spk_id] if enroll_data[0] != spkwav_id]
        enroll_id, enroll_path, enroll_length = random.choice(available_enrolls)
        enrollwav = self._load_wav(enroll_path, enroll_length)
        enroll = torch.from_numpy(enrollwav)
        aux_len = enrollwav.shape[0]

        other_noise, other_id = self._generate_noise(spk_id, 'other')
        mixture = self._mix([spkwav,other_noise])
        if np.max(np.abs(mixture)) >= 1:
            mixture = mixture * self.MAX_AMP / np.max(np.abs(mixture))
        mixture = torch.from_numpy(mixture)

        return mixture, source, enroll, aux_len, spk_id

    def get_infos(self):
        return self.base_dataset.get_infos()

