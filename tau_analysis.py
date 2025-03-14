"""
@author: Kate

2025-03-14
Find average radiation signal by triggering from Linac sync.
"""

import numpy as np
import pandas as pd

def trigger(x, y, threshold_y, refresh_x):
    """Returns the indices of the rising edges in the signal which are above a given threshold and at least "refresh" indices apart in the array"""
    high_i = np.where(y > threshold_y)[0]  # indices of values that are above the threshold
    high_x = np.array(x[high_i]) # times where signal is above the threshold

    rising_i = np.where(np.diff(high_x) > refresh_x)[0] + 1 # indices of "high_x" where previous high value is more than refresh rate away
    trig_i = high_i[rising_i]
    trig_x = x[trig_i]  # subset of high_x that is only the rising times
    trig_y = y[trig_i]  # subset of signal that is only at the rising times

    return trig_i, trig_x, trig_y

def average_pulses(signal_df, threshold_y, refresh_x, window, lead):
    trig_i, trig_x, trig_y = trigger(signal_df['time'], -signal_df['sync_trig'], threshold_y, refresh_x)
    n_pulses = len(trig_i)

    start = round(window / lead)  # number of points before trigger
    end = window - start  # number of points after trigger

    pulses_time = np.zeros((n_pulses, window))
    pulses_trig = np.zeros((n_pulses, window))
    pulses_sig = np.zeros((n_pulses, window))
    pulses_linout = np.zeros((n_pulses, window))

    for pulse_n, pulse_index in enumerate(trig_i):
        pulses_time[pulse_n] = signal_df['time'][pulse_index - start:pulse_index + end] - signal_df['time'][pulse_index] # time relative to pulse trigger
        pulses_trig[pulse_n] = signal_df['sync_trig'][pulse_index - start:pulse_index + end]
        pulses_sig[pulse_n] = signal_df['sig_ac'][pulse_index - start:pulse_index + end]
        pulses_linout[pulse_n] = signal_df['lin_out'][pulse_index - start:pulse_index + end]

    mean_time = np.mean(pulses_time, axis=0)
    mean_signal = np.mean(pulses_sig, axis=0)
    mean_linout = np.mean(pulses_linout, axis=0)
    mean_trigger = np.mean(pulses_trig, axis=0)

    return pulses_time, mean_time, pulses_trig, mean_trigger, pulses_sig, mean_signal, pulses_linout, mean_linout


if __name__ == "__main__":
    """ Example of how to use the functions """
    import matplotlib.pyplot as plt

    # import the csv
    filename1 = '20250312_04_10MVfff.csv'
    sig1 = pd.read_csv(filename1, header=0, names=['time', 'sig_ac', 'lin_out', 'sync_trig'], skiprows=[1,2], dtype=np.float64)

    # remove unwanted channels and convert all units to V
    sig1['sig_ac'] = sig1['sig_ac'] / 1000  # mV to V
    sig1['lin_out'] = sig1['lin_out'] / 1000  # mV to V
    sig1['time'] = sig1['time'] * 1000  # ms to us

    pulses_time, mean_time, pulses_trig, mean_trigger, pulses_sig, mean_signal, pulses_linout, mean_linout \
        = average_pulses(sig1, threshold_y=0.1, refresh_x=2000, window=5000, lead=3)
    n_pulses=pulses_time.shape[0]

    # Plot
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})

    ax[0].plot(pulses_time.T, (pulses_sig.T), 'b', alpha=0.1)
    ax[0].plot(mean_time, mean_signal, 'k')
    ax[0].set_title(f'Averaged AC signal ({n_pulses} pulses)')

    ax[1].plot(pulses_time.T, (pulses_linout.T), 'purple', alpha=0.1)
    ax[1].plot(mean_time, mean_linout, 'k')
    ax[1].set_title(f'Linear output ({n_pulses} pulses)')

    ax[2].plot(pulses_time.T, pulses_trig.T, 'r', alpha=0.1)
    ax[2].plot(mean_time, mean_trigger, 'k')
    ax[2].set_title(f'Sync trigger ({n_pulses} pulses)')
    ax[2].set_xlabel('Time (µs)')

    plt.setp(ax, xlim=(-100, 100))

    fig.supylabel('Signal (V)')
    fig.suptitle(filename1)