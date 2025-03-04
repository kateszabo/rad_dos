"""
@author: Kate

2025-02-26
Kate's version of PSD calculation and plot code. Adapted from Fernanda's code "PSD new.py"
"""

import numpy as np
import pandas as pd
import spinmob as sm

def convert_to_si(databox):
    """Convert loaded data to SI units"""
    units = [databox.hkeys[-1]] + databox.h(-1).split(' ')  # time + channels
    for column, unit in enumerate(units):
        if 'm' in unit:  # ms or mV
            databox[column] = databox[column] / 1e3
            print(f"Converted {column} from {unit} to SI (x1e3)")
        elif 'u' in unit:  # us or uV
            databox[column] = databox[column] / 1e6    
            print(f"Converted {column} from {unit} to SI (x1e6)")    
    return databox

def load_files(name: str):
    """Loads files using SpinMob"""
    print(f"Select {name} files")
    paths = sm.dialogs.load_multiple()
    files = sm.data.load_multiple(paths)
    databoxes_list = [convert_to_si(file) for file in files]
    print(f"Each file contains {len(databoxes_list[0][0])} timesteps")

    folder_name = paths[0].split('/') [-2] # named based on first file in the list

    return databoxes_list, folder_name

def average_psd(databox_list, col):
    """Note: assumes time is in column 0"""
    sum_psd = 0
    for databox in databox_list:  # vectorize this somehow?
        f, psd = sm.fun.psd(databox[0], databox[col])
        sum_psd += psd
    avg_psd = sum_psd / len(databox_list)
    return f, avg_psd

def compute_psds(databox_list, channels):
    """Calculates PSDs for signal and detector data, and coarsens."""
    coarsen_level = 1.01

    freq = [None] * len(channels)
    avg_psd = [None] * len(channels)

    for i, channel in enumerate(channels): 
        f, psd = average_psd(databox_list, channel)
        freq[i], avg_psd[i] = sm.fun.coarsen_data(f, psd, level=coarsen_level, exponential=True)
    
    return freq, avg_psd

def compute_mean_power(signal, eta, gain):
    """signal: 2d array signal[acquisiton file][timestamp]"""

    # compute mean statistical values across all acquisition files
    mean = np.mean(np.mean(signal, axis=1))
    std = np.mean(np.std(signal, axis=1))
    rms = np.mean(np.sqrt(np.mean(signal ** 2, axis=0)))

    print(f"Mean {mean:.2f} V")
    print(f"Standard deviation {std:.2f} V")
    print(f"RMS {rms:.2f} V")

    pwr = mean / (eta * gain)  # V to W
    print(f"Average power {pwr:.4f} W")

    return mean, pwr

def compute_rin(signal_psd, mean_signal, detector_psd):
    """ 
    signal: averaged PSD as array
    detector: averaged PSD as array"""
    rin = np.abs((signal_psd - detector_psd) / mean_signal ** 2)
    return rin

def do_everything(channels: pd.DataFrame):
    # get detector files and PSDs
    detector, detector_filename = load_files("detector")
    f_det, avg_psd_det = compute_psds(detector, channels['column_index'])

    # get signal files and PSDs
    signals, signal_filename = load_files("signal")
    f_sig, avg_psd_sig = compute_psds(signals, detector, channels)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Default settings
    ETA = 0.43  # detector responsivity [A/W]

    GAIN_SIG = 1e4  # detector gain of signal
    GAIN_LINOUT = 1e5  # detector gain of linear output

    channels = [1, 2, 3] 
    channel_names = {1: 'SigDC', 2: 'SigAC', 3: 'LinOut'}

    # load signal and detector files
    signals, detector, folder_name = load_files()

    # compute PSDs
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

    # Calculate mean power
    sig_dc = np.array([signal[1] for signal in signals])
    mean_dc, power_dc = compute_mean_power(sig_dc, ETA, GAIN_SIG)
    mean_linout = mean_dc * GAIN_LINOUT / GAIN_SIG

    # Calculate RIN = (signal - detector) / (V^2)
    rin_ac = compute_rin(avg_psd_sig[0], mean_dc, avg_psd_det[0])
    rin_linout = compute_rin(avg_psd_sig[2], mean_linout, avg_psd_det[2])

    # Plot RIN
    fig, ax = plt.subplots()
    ax.loglog(f_sig[0], rin_ac, label='Signal (AC)')
    ax.loglog(f_sig[0], rin_linout, label='Linear Output')

    fig.suptitle('Laser noise (13 Feb)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('RIN (1/Hz)')
    ax.set_xlim(0, 1e7)
    ax.set_ylim(1e-16, 1e-9)
    ax.legend()