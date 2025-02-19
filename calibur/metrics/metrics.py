#!/usr/bin/env python
# https://github.com/csteinmetz1/auraloss/blob/main/auraloss/time.py

import numpy as np
from pystoi import stoi

eps = 1e-8

# def sisdr(x, s, remove_dc=True):
#     """
#     Compute SI-SDR
#     x: extracted signal
#     s: reference signal(ground truth)
#     """

#     def vec_l2norm(x):
#         return np.linalg.norm(x, 2)

#     if remove_dc:
#         x_zm = x - np.mean(x)
#         s_zm = s - np.mean(s)
#         t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
#         n = x_zm - t
#     else:
#         t = np.inner(x, s) * s / vec_l2norm(s)**2
#         n = x - t
#     return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


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