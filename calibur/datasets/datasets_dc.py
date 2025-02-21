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
from torchaudio.transforms import Resample
import torchaudio
import torch as th
import warnings
import pdb

# def read_enrollment_csv(csv_path):
#     data = defaultdict(dict)
#     with open(csv_path, 'r') as f:
#         f.readline() # csv header

#         for line in f:
#             mix_id, utt_id, *aux = line.strip().split(',')
#             aux_it = iter(aux)
#             aux = [(auxpath,int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
#             data[mix_id][utt_id] = aux
#     return data

class LibriMixInformed_dc(Dataset):
    def __init__(
        self, csv_dir, utt_scp_file,noise_scp_file, spk_list, task="sep_clean", sample_rate=16000, n_src=2, 
        MAX_AMP=0.9, MIN_LOUDNESS=-33, MAX_LOUDNESS=-25, mean_snr=-2, var_snr=15,segment=3, segment_aux=3, 
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
        self.noise, _ = self._load_data(noise_scp_file, is_spk=False)
        self.speeds = [0.8, 0.9, 1.0, 1.1, 1.2]
    
        self.total_lines = len(self.data)

        self.meter = pyloudnorm.Meter(self.sample_rate)
        self.MAX_AMP = 0.9
        self.MIN_LOUDNESS = -33
        self.MAX_LOUDNESS = -25
        self.mean_snr = -2
        self.var_snr = 15

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

    def _snr_norm(self, signal, noise, is_noise=True):
        if is_noise:
            desired_snr = np.random.normal(self.mean_snr, self.var_snr**0.5)  
            current_snr = 10 * np.log10(np.mean(signal ** 2) / (np.mean(noise ** 2) + self.EPS) + self.EPS)
            scale_factor = 10 ** ((current_snr - desired_snr ) / 20)
            scaled_noise = noise * scale_factor
        return scaled_noise

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


    def _datahandler(self, wav, length, drop=0):
        if length <= self.comm_length:
            # wav,_ = sf.read(path, dtype="float32")
            # if len(wav.shape) != 1:
            #     wav = wav[:, np.random.randint(0, wav.shape[1])]
            padding_length = self.comm_length - length
            #  np.pad  0
            wav = np.pad(wav, (0, padding_length), 'constant')
        elif length > self.comm_length:
        # start to random start
            start, stop = self._get_segment_start_stop(self.comm_length, length, drop)
            wav = wav[start:stop]

        # if len(wav.shape) != 1:
        #     wav = wav[:, -1]  
        else:
            print("error length:",length)

        return wav

    def _generate_noise(self,spk_id,type):
        # just other spk
        if type == 'other':
            other_spks = [spk for spk in self.data_spk.keys() if spk != spk_id]
            random_spk = np.random.choice(other_spks, 1)[0]
            element = random.choice(self.data_spk[random_spk])
            noise_id = element[0]
            path = element[1]
            length = element[2]
        # noise = self._load_wav(path, length)   
        elif type == 'noise':
            element = random.choice(self.noise)
            noise_id = element[0]
            path = element[1]
            length = element[2]

        return noise_id, path, length
  
    def _add_speed_perturb(self, waveform, enroll, label):
        # init resamplers
        resamplers = []

        for speed in self.speeds:
            config = {
                "orig_freq": self.sample_rate,
                "new_freq": self.sample_rate * speed,
            }
            resamplers.append(Resample(**config))

        wav_len = waveform.shape[0]
        # Convert numpy arrays to PyTorch tensors
        waveform = th.from_numpy(waveform).float()
        enroll = th.from_numpy(enroll).float()

        samp_index = th.randint(0, len(self.speeds), (1,)).item()

        waveform = resamplers[samp_index](waveform)
        enroll = resamplers[samp_index](enroll)
        label = label + self.num_spk * samp_index

        eol, _ = torchaudio.sox_effects.apply_effects_tensor(
            enroll.unsqueeze(0), self.sample_rate,
            [['tempo', str( self.speeds[samp_index])], ['rate', str(self.sample_rate)]])
        perturbed_enroll = eol.numpy()[0]
        
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform.unsqueeze(0), self.sample_rate,
            [['tempo', str( self.speeds[samp_index])], ['rate', str(self.sample_rate)]])

        perturbed_waveform = wav.numpy()[0]
        # Convert tensors back to numpy arrays if needed (e.g., for further processing)

        #self.comm_length = perturbed_waveform.shape[0]

        #assert perturbed_waveform.shape[0] == wav_len, f"Length mismatch: perturbed_waveform length is {perturbed_waveform.shape[0]}, expected {wav_len}"

        return perturbed_waveform, perturbed_enroll, label

    def _normalize(self, signal):
        c_loudness = self.meter.integrated_loudness(signal)

        target_loudness = random.uniform(self.MIN_LOUDNESS, self.MAX_LOUDNESS)  
        # print("wav lound:",target_loudness)


        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # 捕获所有警告
            signal = pyloudnorm.normalize.loudness(signal, c_loudness, target_loudness)

            # 检查警告并处理
            if w:
                for warning in w:
                    pass    
        return signal

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
        spkwav,_ = sf.read(spkwav_path, dtype="float32")

        available_enrolls = [enroll_data for enroll_data in self.data_spk[spk_id] if enroll_data[0] != spkwav_id]
        enroll_id, enroll_path, enroll_length = random.choice(available_enrolls)
        enrollwav,_ = sf.read(enroll_path, dtype="float32")

        spkwav, enrollwav, spk_id = self._add_speed_perturb(spkwav, enrollwav, spk_id)
        spkwav_length = spkwav.shape[0]
        enroll_length = enrollwav.shape[0]

        other_noise_id, other_noise_path, other_noise_length = self._generate_noise(spk_id,'other')
        other_noise,_ = sf.read(other_noise_path, dtype="float32")

        noise_id, noise_path, noise_length = self._generate_noise(spk_id,'noise')
        noise,_ = sf.read(noise_path, dtype="float32")


        spkwav = self._normalize(spkwav)
        enrollwav = self._normalize(enrollwav)


        spkwav = self._datahandler(spkwav, spkwav_length)
        enrollwav = self._datahandler(enrollwav, enroll_length)
        other_noise = self._datahandler(other_noise, other_noise_length)
        noise = self._datahandler(noise, noise_length)

        aux_len = enrollwav.shape[0]


        cm_noise = self._mix([other_noise,noise])

        cm_noise = self._snr_norm(spkwav, cm_noise)

        mixture = self._mix([spkwav,cm_noise])

        if np.max(np.abs(mixture)) >= 1:
            mixture = mixture * self.MAX_AMP / np.max(np.abs(mixture))


        source = torch.from_numpy(spkwav)[None]
        mixture = torch.from_numpy(mixture)
        enroll = torch.from_numpy(enrollwav)

        return mixture, source, enroll, aux_len, spk_id

    def get_infos(self):
        return self.base_dataset.get_infos()

