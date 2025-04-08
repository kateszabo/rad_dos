""" 
Copy of Fernanda's code that can be imported as a module.
Created by Kate on 2025-04-08
"""
import numpy as np
import spinmob as sm
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal

np.NAN = np.nan  # catch numpy errors with NAN

SIGdata = '53'  # folder number

detector = 'PDA10A'
signal_gain  = 5e3     # detector transimpedance gain in V/A
responsivity = 0.403  # detector responsivity in A/W

save_results = 0
save_figures = 0

#%%     # define functions

def load_data(SIGdata):
    """
    Load data from the specified folder.
    """
    # Load all .txt files in the specified folder
    file_list = glob.glob(f'*_{SIGdata}_*/*.txt')
    
    # store data
    data     = sm.data.load_multiple(file_list)
    if '/' in file_list[0]: labels = file_list[0].split('/')[-2]
    else:                   labels = file_list[0].split('\\')[-2]
    date     = labels.split('_')[0]
    channels = (' Channel Time '+ data[0].h('Time')).split(' Channel ')[1:]
    
    return file_list, data, labels, date, channels

def convert_to_binary(data):
    """
    Check if data is already in binary format.
    If not, convert and save in binary format.
    """
    for i,d in enumerate(data):
        if not 'float' in d.h(0):
            if os.path.exists( file_list[i] ): os.remove( file_list[i] )    # remove the file if it already exists
            d.save_file( file_list[i], binary=True)
            print(f'Converting {d} to binary format.')
            ds[i] = sm.data.load( file_list[i])
        else:
            print(f'{d} is already in binary format.')

def fix_units(data):

    for i, d in enumerate(data):
        units = [ d.hkeys[-1] ] + d.h(-1).split(' ')
        for ch, unit in enumerate(units):
            if 'm' in unit: 
                d[ch] = d[ch] * 1e-3
                if i == 0: print(f'Converted ch{ch} units from {unit} to SI')
            elif 'u' in unit:
                d[ch] = d[ch] * 1e-6
                if i == 0: print(f'Converted ch{ch} units from {unit} to SI')

def average_pulse(data):
    pulse_avg = np.mean( data, axis=0 )
    pulse_err = np.std ( data, axis=0 )/ np.sqrt(len(data)-1) # find the error on the mean
    pulse_rms = np.sqrt( np.mean( np.square(data), axis=0 ) )

    avgDC = np.mean( pulse_avg[1][:len(pulse_avg[1])//2] )
    avgP  = avgDC/(signal_gain*responsivity)  # average power in W
    
    return pulse_avg, pulse_err, pulse_rms, avgDC, avgP

def averaged_PSD(data, t_channel, y_channel):
    """
    Calculate the averaged Power Spectral Density (PSD) of the data.
    """
    sum=0
    for acq in range(len(data)):
        f, psd = sm.fun.psd( data[-1][t_channel], data[acq][y_channel])
        sum += psd
    
    PSD = sum/len(data)
    f, PSD = sm.fun.coarsen_data(f, PSD, level=1.01, exponential=True)
    return f, PSD

def compute_PSD(data):
    F, P = [ sm.data.databox() for i in range(2) ]

    for i, channel in enumerate(channels[1:]):
        F, P[channel] = averaged_PSD(data, 0, i+1)

    return F, P

def compute_RIN(channel, gain_factor, P, eta=responsivity, echarge=1.6e-19):
    RIN = sm.data.databox()
    for i, ch in enumerate(channel):
        RIN[ch] = P[ch] / (avgDC * gain_factor[i])**2

    RIN_SN = 2 * echarge / (eta * avgP)  # relative shot noise floor

    return RIN, RIN_SN

def integrate_PSD(channel, bandwidth):
    rms_psd= sm.data.databox()
    rms_rin= sm.data.databox()
    points_in_bandwidth = np.where( F <= bandwidth )[0][-1] # find the index of the last point in the bandwidth
    
    for ch in channel:
        bins_psd = 0
        bins_rin = 0
    
        for point in range ( points_in_bandwidth ):        
            bins_psd +=   P[ch][point] * np.diff( F )[point]
            bins_rin += RIN[ch][point] * np.diff( F )[point]

        rms_psd[ch] = [np.sqrt(bins_psd)]
        rms_rin[ch] = [np.sqrt(bins_rin)]
     
        print ( f'{ch} min detectable change in V = {rms_psd[ch][0]:.2e} at {bandwidth:.2e} Hz bw' )
    print()

    return rms_psd, rms_rin, bandwidth

def save_averaged_data(channel):
    ds_averaged = sm.data.databox()
    ds_averaged.h( signal_data = labels, photodiode = detector, gain_SIG = signal_gain,  
                   avg_DCvoltage = avgDC, avg_P = avgDC/(signal_gain*responsivity), channels = channels)

    for i,ch in enumerate(channel):
        ds_averaged[ch] = pulse_avg[i]
        ds_averaged[ch+'_err'] = pulse_err[i]

    directory = os.getcwd() + '/avgs'
    if not os.path.exists(directory): os.makedirs(directory)

    save_path = f'./avgs/{labels}_AVG_PULSE.txt'
    if os.path.exists(save_path): os.remove(save_path)  # remove the file if it already exists
    ds_averaged.save_file( save_path )

def save_PSD_data(channel):
    ds_psd = sm.data.databox()
    ds_psd.h( signal_data = labels, photodiode = detector, gain_SIG = signal_gain, 
              avg_DCvoltage = avgDC, avg_P = avgP,
              rms_psd = rms_psd, rms_rin = rms_rin, integration_bandwidth = integration_bandwidth,
              channels = channel)

    ds_psd['Frequency'] = F
    for ch in channel:
        ds_psd[ch] = P[ch]

    directory = os.getcwd() + '/psds'
    if not os.path.exists(directory): os.makedirs(directory)

    save_path = f'./psds/{labels}_PSD.txt'
    if os.path.exists(save_path): os.remove(save_path)  # remove the file if it already exists
    ds_psd.save_file( save_path )

def save_RIN_data(channel):
    ds_rin = sm.data.databox()
    ds_rin.h( signal_data = labels, photodiode = detector, gain_SIG = signal_gain,
              avg_DCvoltage = avgDC, avg_P = avgP,
              rms_psd = rms_psd, rms_rin = rms_rin, integration_bandwidth = integration_bandwidth,
              shotnoise = RIN_SN, channels = channels)

    ds_rin['Frequency'] = F
    for ch in channel:
        ds_rin[ch] = RIN[ch]

    directory = os.getcwd() + '/rin'
    if not os.path.exists(directory): os.makedirs(directory)

    save_path = f'./rin/{labels}_RIN.txt'
    if os.path.exists(save_path): os.remove(save_path)  # remove the file if it already exists
    ds_rin.save_file( save_path )

def find_pulse_on(pulse_avg, pulse_err, T, TRIG):
    """ Get the time range (in us) over which the radiation pulse was on.
    """
    t_pulse_on = pulse_avg[T][ pulse_err[TRIG]>0.25 * max( pulse_err[TRIG]) ] * 1e6
    t0_pulse = t_pulse_on[0]
    dt_pulse = t_pulse_on[-1] - t_pulse_on[0] 
    return t_pulse_on, t0_pulse, dt_pulse

def lowpass_filter(t, v, f_lowpass, fs=None):
    """
    Apply a causal low-pass filter to time-domain voltage data.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Array of time values corresponding to voltage measurements
    v : numpy.ndarray
        Array of voltage measurements to be filtered
    f_lowpass : float
        Cutoff frequency of the low-pass filter in Hz
    fs : float, optional
        Sampling rate of the data. If None, calculated from time array
    
    Returns:
    --------
    numpy.ndarray
        Filtered voltage array
    """
    # Calculate sampling rate if not provided
    if fs is None:
        fs = 1 / (t[1] - t[0])
    
    # Nyquist frequency
    nyquist = 0.5 * fs
    
    # Normalize the cutoff frequency
    normalized_cutoff = f_lowpass / nyquist
    
    # Design low-pass filter
    b, a = signal.butter(N=1, Wn=normalized_cutoff, btype='low', analog=False)
    
    # Apply causal filter (forward only)
    filtered_v = signal.lfilter(b, a, v)
    
    return filtered_v

def pulse(t, t_rad, tau):
    """
    Function describing exponential rise for duration t_rad and exponential decay afteward.

    Parameters
    ----------
    t : numpy.ndarray
        Time array. Must be numpy array to work with filter.
    t_rad : float
        How long the pulse rises for.
    tau : float
        Time constant for rise and fall.    
    """
    # Calculate the voltage value for each point, vectorized so we can handle arrays
    v = np.zeros_like(t)
    v[(t >= 0) & (t <= t_rad)] = 1-np.exp(-t[(t >= 0) & (t <= t_rad)]/tau)
    v[t > t_rad] = (1-np.exp(-t_rad/tau))*np.exp(-(t[t > t_rad]-t_rad)/tau)
    return v

def pulse_filtered(t, t_rad, tau, f_lowpass=500e3):
    """
    Low-passed version of the pulse above.

    Parameters
    ----------
    t : numpy.ndarray
        Time array. Must be numpy array to work with filter.
    t_rad : float
        How long the pulse rises for.
    tau : float
        Time constant for rise and fall.
    f_lowpass : float
        Low-pass filter frequency [Hz].
    """
    v = pulse(t, t_rad, tau)
    return lowpass_filter(t, v, f_lowpass)

if __name__ == "__main__":

    if save_figures:
        if not os.path.exists(os.getcwd() + '/plots'): 
            os.makedirs(os.getcwd() + '/plots')

    # %%    # load data, convert to binary, and fix axes units
    file_list, ds, labels, date, channels = load_data(SIGdata)
    convert_to_binary(ds)
    fix_units(ds)

    # %%    # average the acquisitions, compute PSD, and compute RIN
    pulse_avg, pulse_err, pulse_rms, avgDC, avgP = average_pulse(ds)
    F, P = compute_PSD(ds)
    RIN, RIN_SN  = compute_RIN(['C_SIGac'], [1], P)

    rms_psd, rms_rin, integration_bandwidth = integrate_PSD(['C_SIGac'], 200e3)

    # %%   # save the averaged data, PSD, and RIN
    if save_results:
        save_averaged_data(channels)
        save_PSD_data(['C_SIGac'])
        save_RIN_data(['C_SIGac'])

    #%%     # close figures
    close=1
    if close:
        plt.close('all')
    # %%    # plot settings
    for i, ch in enumerate(channels):
        if     'Time' in ch:    T = i
        elif 'SIGdc'  in ch:   DC = i
        elif 'SIGac'  in ch:   AC = i
        elif 'REF'    in ch:  REF = i
        elif 'LINAC'  in ch: TRIG = i
        
    myc = ['#ffc337','k','#d1cdc2','#713A67','#a7c4ae','#FF5B5C', '#326d9d', '#289ced', '#c488db','#e8338e']
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler('color', myc)
        ,'axes.formatter.useoffset' : False
        ,'axes.labelpad'        : 1.5
        ,'axes.linewidth'       : 1
        ,'figure.constrained_layout.use': True
        ,'figure.figsize'       : (7.5,2.5) #(4, 2.95) #(6.5,4.4)
        ,'figure.titlesize'     : 10
        ,'font.size'            : 11
        ,'legend.borderaxespad' : 0.7
        ,'legend.borderpad'     : 0.1
        ,'legend.edgecolor'     : 'white'
        ,'legend.fancybox'      : False
        ,'legend.fontsize'      : 11
        ,'legend.framealpha'    : .8
        ,'legend.frameon'       : True
        ,'legend.handlelength'  : 1.0
        ,'legend.handletextpad' : 0.2
        ,'legend.labelspacing'  : 0.2
        ,'legend.markerscale'   : 1.5
        ,'legend.numpoints'     : 1
        ,'lines.dashed_pattern' : [2, 2]
        ,'lines.linewidth'      : 1.5
        ,'lines.markeredgecolor': 'auto'
        ,'lines.markeredgewidth': 0.5
        ,'lines.markerfacecolor': 'auto'
        ,'lines.markersize'     : 6.0
        ,'mathtext.fontset'     : 'dejavuserif' #'cm' #'dejavusans' #'stixsans'
        ,'xtick.major.pad'      : 4
        ,'ytick.major.pad'      : 2.5
        ,'xtick.major.size'     : 6
        ,'ytick.major.size'     : 6
        ,'xtick.minor.size'     : 4
        ,'ytick.minor.size'     : 4

        ,'xtick.major.width'    : 1
        ,'ytick.major.width'    : 1
        ,'ytick.major.right'    : True
        ,'ytick.minor.right'    : True

        ,'axes.titlesize'       : 'medium'
        ,'axes.labelsize'       : 11  # fontsize of the x and y labels
        ,'xtick.labelsize'      : 11  # fontsize of the tick labels
        ,'ytick.labelsize'      : 11  # fontsize of the tick labels

        ,'savefig.directory'    : '~/Desktop/'
        ,'savefig.dpi'          : 200.0
        ,'savefig.format'       : 'png'
        ,'savefig.transparent'  : False
        # ,'text.latex.preamble': ''
        # ,'text.usetex': True
        })

    # %%    # plot the averaged pulse
    plt.figure()
    plt.plot(pulse_avg[T] * 1e6, pulse_avg[AC] * 1e3, label=channels[AC] )
    plt.fill_between(
        pulse_avg[T] * 1e6,
        (pulse_avg[AC] - pulse_err[AC]) * 1e3,
        (pulse_avg[AC] + pulse_err[AC]) * 1e3,
        alpha=0.1
    )

    plt.xlabel('Time [us]')
    plt.ylabel('Voltage [mV]')
    plt.title(f'{labels} -- Averaged Pulse ({len(ds)} acquisitions), avg P={avgP*1e6:.2f} uW')
    plt.legend( loc='lower right')
    plt.xlim(-100, pulse_avg[T][-1]*1e6 )

    ax2 = plt.gca().twinx()
    ax2.plot(pulse_avg[T] * 1e6, pulse_avg[TRIG] * 1e3, myc[6])
    ax2.fill_between(
        pulse_avg[T] * 1e6,
        (pulse_avg[TRIG] - pulse_err[TRIG]) * 1e3,
        (pulse_avg[TRIG] + pulse_err[TRIG]) * 1e3,
        alpha=0.1
    )
    ax2.set_ylabel('Trigger (mV)', color=myc[6])  
    ax2.tick_params(axis='y', labelcolor=myc[6], pad=2.5, color=myc[6])
    ax2.tick_params(which='minor', color=myc[6])
    ax2.spines['right'].set_color(myc[6])

    if save_figures:    plt.savefig(f'./plots/{labels}_AVG_PULSE')

    # %%    # plot the averaged PSD
    fig, ax = plt.subplots(1, len(channels[1:-1]), figsize=(9,3), sharex=True, sharey=True)
    plt.suptitle(f'{labels} -- Averaged PSD ({len(ds)} acquisitions), avg P={avgP*1e6:.2f} uW')

    for i, channel in enumerate(channels[1:-1]):
        print(i,channel)
        ax[i].loglog(F, P[channel], label=labels)
        
        ax[i].set_xlabel('Frequency [Hz]')
        ax[i].set_ylabel( f'{channel} PSD [V^2/Hz]')
        
        ax[i].set_xlim(F[0], F[-1])
        # ax[i].set_ylim(bottom = 1e-15)
        ax[i].tick_params(axis='y', labelleft=True)

    ax[0].legend()
    ax[1].legend(['dV = {:.2e} V'.format(rms_psd['C_SIGac' ][0])])

    if save_figures:    plt.savefig(f'./plots/{labels}_PSD')

    # %%    # plot the RIN
    plt.figure()
    plt.loglog(F, RIN['C_SIGac' ],         label='C_SIGac')

    plt.axhline(y=RIN_SN,   color=myc[0], linestyle='--', label='Shot noise floor')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('RIN [1/Hz]')
    plt.xlim(F[0], F[-1])
    plt.legend()
    plt.title(f'{labels} -- RIN ({len(ds)} acquisitions), avg P={avgP*1e6:.2f} uW')

    if save_figures:    plt.savefig(f'./plots/{labels}_RIN')

    # %%
    print(labels)


    # %%    # fit the averaged pulse

    plt.rcParams.update({'figure.figsize': (7,5)})
    t_pulse_on, t0_pulse, dt_pulse = find_pulse_on(pulse_avg, pulse_err, T, TRIG)

    # FIT DATA TO FILTERED MODEL
    f = sm.data.fitter()
    f.set_data( pulse_avg[T] * 1e6, pulse_avg[AC] * 1e6, 10)
    f.set_functions('A * pf( x-t0,dt,tau,flp ) + a + b*x + c*x**2', 
                    'A=1e3, t0=0, tau=10, a=0,b=0,c=0',  
                    bg='a+b*x+c*x**2',
                    g=dict(dt=dt_pulse, flp=0.125, pf=pulse_filtered))
    f.set(coarsen=12,
            plot_guess=False, 
            xlabel='Time (us) -- '+labels, 
            ylabel='Normalized Power Change (ppm)',
            xmin=-50, xmax=200)
    f.fit().autoscale_eydata().fit()
    # plt.ginput()

    pulse_relative_amp = (  f.p_fit['A'].value* (1-np.exp(-dt_pulse/f.p_fit['tau'].value))
                        /(f.p_fit['a'].value+avgDC*1e6) )

    print( f'pulse relative amplitude = {pulse_relative_amp*1e6:.0f} ppm')
    if f.p_fit["tau"].stderr is not None:
        print( f'tau = {f.p_fit["tau"].value:.2f} +/- {f.p_fit["tau"].stderr:.2f} us')

    # %%

