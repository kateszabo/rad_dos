"""
@author: Kate

2025-02-26
Kate's version of PSD calculation and plot code. Adapted from Fernanda's code "PSD new.py"
"""

import numpy as np
import spinmob as sm

def convert_to_si(databox):
    units = [databox.hkeys[-1]] + databox.h(-1).split(' ')  # time + channels
    for column, unit in enumerate(units):
        if 'm' in unit:  # ms or mV
            databox[column] = databox[column] / 1e3
            # print(f"Converted {column} from {unit} to SI (x1e3)")
        elif 'u' in unit:  # us or uV
            databox[column] = databox[column] / 1e6    
            # print(f"Converted {column} from {unit} to SI (x1e6)")    
    return databox

def load_files():
    # Load files
    print("Select signal files")
    sig_paths = sm.dialogs.load_multiple()
    sig_data = sm.data.load_multiple(sig_paths)
    signals = [convert_to_si(file) for file in sig_data]

    print("Select detector files")
    det_paths = sm.dialogs.load_multiple()
    det_data = sm.data.load_multiple(det_paths)
    detector = [convert_to_si(file) for file in det_data]

    folder_name = sig_paths[0].split('/') [-2] # named based on first file in signal data

    return signals, detector, folder_name

def average_psd(databox_list, col):
    """Note: assumes time is in column 0"""
    sum_psd = 0
    for databox in databox_list:
        f, psd = sm.fun.psd(databox[0], databox[col])
        sum_psd += psd
    avg_psd = sum_psd / len(databox_list)
    return f, avg_psd

def compute_psds(signals, detector, channels):
    f_sig = [None] * len(channels)
    avg_psd_sig = [None] * len(channels)

    f_det = [None] * len(channels)
    avg_psd_det = [None] * len(channels)

    for i, channel in enumerate(channels): 
        f_sig[i], avg_psd_sig[i] = average_psd(signals, channel)
        f_det[i], avg_psd_det[i] = average_psd(detector, channel)

    return f_sig, avg_psd_sig, f_det, avg_psd_det


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    signals, detector, folder_name = load_files()

    channels = [1, 2, 3] 
    channel_names = {1: 'SigDC', 2: 'SigAC', 3: 'LinOut'}
    f_sig, avg_psd_sig, f_det, avg_psd_det = compute_psds(signals, detector, channels)

    # plot each channel (signal and detector)
    fig, ax = plt.subplots(1, len(channels), sharey=True)

    for i, channel in enumerate(channels):
        ax[i].loglog(f_sig[i], avg_psd_sig[i], label="Signal")
        ax[i].loglog(f_det[i], avg_psd_det[i], label="Detector")
        ax[i].loglog()
        ax[i].set_title(channel_names[channel])
        ax[i].set_xlabel("Frequency (Hz)")
        
    fig.suptitle(f"PSDs for {folder_name}")
    ax[0].set_ylabel(r'PSD ($\frac{V}{\sqrt{Hz}}$)')
    ax[-1].legend()

    plt.show()
