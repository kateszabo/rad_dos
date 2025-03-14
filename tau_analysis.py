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

def average_pulses(signal_df, channel_key, threshold_y, refresh_x, window, lead):
    trig_i, trig_x, trig_y = trigger(signal_df['time'], -signal_df['sync_trig'], threshold_y, refresh_x)
    n_pulses = len(trig_i)

    start = round(window / lead)  # number of points before trigger
    end = window - start  # number of points after trigger

    pulses_time = np.zeros((n_pulses, window))
    pulses_trig = np.zeros((n_pulses, window))
    pulses_sig = np.zeros((n_pulses, window))

    for pulse_n, pulse_index in enumerate(trig_i):
        pulses_time[pulse_n] = signal_df['time'][pulse_index - start:pulse_index + end] - signal_df['time'][pulse_index] # time relative to pulse trigger
        pulses_trig[pulse_n] = signal_df['sync_trig'][pulse_index - start:pulse_index + end]
        pulses_sig[pulse_n] = signal_df[channel_key][pulse_index - start:pulse_index + end]

    mean_time = np.mean(pulses_time, axis=0)
    mean_signal = np.mean(pulses_sig, axis=0)
    mean_trigger = np.mean(pulses_trig, axis=0)

    return pulses_time, mean_time, pulses_trig, mean_trigger, pulses_sig, mean_signal