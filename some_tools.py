import matplotlib.pyplot as plt
import os
import logging
import torch
import torch.nn.functional as F
import seaborn as sns
import numpy as np

def _logger(file_name='log_files'):
    logger = logging.getLogger()
    if not logger.handlers:
        file_handler = logging.FileHandler(file_name + '.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(file_formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    return logger


def mmd_loss(source, target, sigma=1.0, kernel="linear"):
    def linear_kernel(x, y):
        return torch.mm(x, y.t())
    def gaussian_kernel(x, y, sigma=1.0):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        distance = torch.sum((x - y) ** 2, dim=2)
        kernel = torch.exp(-distance / (2 * sigma ** 2))
        return kernel
    def compute_kernel(x, y, kernel="linear", sigma=1.0):
        if kernel == "linear":
            return linear_kernel(x, y)
        elif kernel == "rbf":
            return gaussian_kernel(x, y, sigma)
        else:
            raise ValueError("Unsupported kernel type: {}".format(kernel))
    k_ss = compute_kernel(source, source, kernel, sigma)
    k_tt = compute_kernel(target, target, kernel, sigma)
    k_st = compute_kernel(source, target, kernel, sigma)
    mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
    return mmd
