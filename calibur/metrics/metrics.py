
import numpy as np
from pystoi import stoi

eps = 1e-8
import torch
import torch.nn.functional as F

# import utmos
# utmos_model = utmos.Score()

# def utmos_score(est,fs):
#     est = torch.tensor(est, dtype=torch.float32)
#     wav = est.unsqueeze(0)
#     mos = utmos_model.calculate_wav(wav, fs)
#     #score = predictor(est, fs)
#     return mos.item()

# from pypesq import pesq


# def compute_pesq(est,ref, fs=16000):
#     if len(est) != len(ref):
#         raise ValueError("Input signals must have the same length")
#     return pesq(ref, est, fs)

def sisdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / (vec_l2norm(s_zm)**2 + eps)
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10( (vec_l2norm(t) / (vec_l2norm(n) + eps)) + eps)

def se_sisdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2 + eps
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / (vec_l2norm(s)**2 + eps)
        n = x - t
    return 20 * np.log10((vec_l2norm(t) + eps)/ (vec_l2norm(n) + eps))


def compute_stoi(x, s, fs=16000):
    """
    Compute STOI
    x: extracted signal
    s: reference signal (ground truth)
    fs: sampling frequency (default: 16000)
    """
    if len(x) != len(s):
        raise ValueError("Input signals must have the same length")
    return stoi(s, x, fs, extended=False)

def compute_speaker_similarity(est, ref, speaker_encoder):
    """
    Compute speaker similarity using cosine similarity of speaker embeddings
    est: extracted signal
    ref: reference signal (ground truth)  
    computer: NnetComputer instance
    """
    y_cand_emb = speaker_encoder.extract_spk_embedding(est)
    aux_emb = speaker_encoder.extract_spk_embedding(ref)

    y_cand_emb = torch.tensor(y_cand_emb, dtype=torch.float32)
    aux_emb = torch.tensor(aux_emb, dtype=torch.float32)
    
    if y_cand_emb.dim() == 1:
        y_cand_emb = y_cand_emb.unsqueeze(0)
    if aux_emb.dim() == 1:
        aux_emb = aux_emb.unsqueeze(0)

    est_emb_norm = F.normalize(y_cand_emb, p=2, dim=1)
    ref_emb_norm = F.normalize(aux_emb, p=2, dim=1) 
    
    similarity = F.cosine_similarity(est_emb_norm, ref_emb_norm, dim=1).item()

    return similarity
