''' Mission: Look at the times of the large range drops and look at the quad 
master_outs, dcpds, and calib_strain to see how the ifo is responding during 
those times. Compare with range drops at LLO '''

import argparse
import os
import time

import gwpy.timeseries as ts
import h5py
import numpy as np
import tqdm
from gwpy.time import tconvert
from scipy import signal

os.environ["NDS2_CLIENT_GAP_HANDLER"] = "STATIC_HANDLER_NEG_INF"
os.environ["NDS2_CLIENT_ALLOW_DATA_ON_TAPE"] = '1'
port = 31200

def startround(num, base=60) -> int:
    '''Rounds number up to the next minute'''
    return int(base * np.ceil(float(num)/base))

def endround(num, base=60) -> int:
    '''Rounds number down to the previous minute'''
    return int(base * np.floor(float(num)/base))

def gps2utc(gps):
    return tconvert(gps).strftime('%m/%d/%y %H:%M:%S')

def stretch2totaltime(obslock_start_end):
    '''Adds up total time locked and observing that we are looking over'''
    total_seconds = 0
    
    for lock in obslock_start_end:
        total_seconds = total_seconds + (lock[1] - lock[0])
    hours = round(total_seconds/3600, 1)
    print(f'Hours Locked and Observing: {hours}')
    return hours

def make_obslock_dict(gpstime, chanval) -> dict:
    dict = {
        'gps': gpstime,        # Starting gps time
        'val': chanval
    }
    return dict

def make_range_dict(gpstime, minval) -> dict:
    dict = {
        'gps': gpstime,        # Starting gps time
        'min': minval
    }
    return dict

def make_etm_dict(gpstime, maxval, minval) -> dict:
    dict = {
        'gps': gpstime,        # Starting gps time
        'max': maxval,
        'min': minval
    }
    return dict

def get_obs_data(start, end):
    '''Grab GRD-IFO state data minute .min trends'''
    
    start = startround(start)
    end = endround(end)
    
    obslock_data = ts.TimeSeriesDict.get(
        [f'{ifo}:GRD-IFO_STATE_N.min,m-trend'],
        start,
        end,
        host=host,
        port=port,
        pad=float('-inf')
    )
    
    obslock_dt = 60
    obslock_size = obslock_data[f'{ifo}:GRD-IFO_STATE_N.min,m-trend'].value.size
    obslock_vals = obslock_data[f'{ifo}:GRD-IFO_STATE_N.min,m-trend'].value
    
    obslock = {}
    for i in range(obslock_size):
        gps = start + (obslock_dt*i)
        obslock[i] = make_obslock_dict(gps, obslock_vals[i])

    # Go through and figure out what times we were locked
    obslock_start_end = []
    for i in range(0, len(obslock)):
        if obslock[i]['val'] == 100:
            if i == 0:
                obslock_start_end.append([obslock[i]['gps']])
            elif obslock[i-1]['val'] != 100:
                obslock_start_end.append([obslock[i]['gps']])
            else:
                continue
        elif obslock[i]['val'] != 100:
            if i == 0:
                continue
            elif obslock[i-1]['val'] != 100:
                continue
            elif obslock[i-1]['val'] == 100:
                obslock_start_end[-1] = [obslock_start_end[-1][0], obslock[i-1]['gps']]
    # If last obslock_start_end section doesn't have a lock end time, add end
    if len(obslock_start_end[-1]) == 1:
        obslock_start_end[-1] = [obslock_start_end[-1][0], end]
    
    return obslock_start_end

def get_range(obslock_start_end, rangedropval):
    '''Grabs range data during lock stretches'''
    
    range_drops = []
    for start_end in obslock_start_end:
        if start_end[1] <= start_end[0]:
            print(f'Skipping zero-duration observing interval: {start_end}')
            continue
        if ifo == 'L1':
            range_chan = [f'{ifo}:CDS-SENSEMON_CAL_{ifo}_SNSW_EFFECTIVE_RANGE_MPC.min,s-trend']
        elif ifo == 'H1':
            range_chan = [f'{ifo}:CDS-SENSMON_CAL_SNSW_EFFECTIVE_RANGE_MPC.min,s-trend']
        # Grab data
        range_data = ts.TimeSeriesDict.get(
                range_chan,
                start_end[0],
                start_end[1],
                host=host,
                port=port,
                pad=float('-inf')
        )
        
        range_dt = range_data[range_chan[0]].dt.value
        range_size = range_data[range_chan[0]].value.size
        range_vals = range_data[range_chan[0]].value
        
        for i in range(1, range_size):  # Remove the first value in case it's weird
            gps = start_end[0] + (range_dt*i)
            range_min = range_vals[i]
            if range_min <= rangedropval:
                # Get rid of -inf in data
                if np.isneginf(range_min):
                    continue
                # Check if last range_min was the same value
                if len(range_drops) > 0:
                    last_range_min = range_drops[-1]['min']
                    if range_min == last_range_min:
                        continue
                range_drops.append(make_range_dict(gps, range_min))
    return range_drops

def check_for_glitch(range_drops):
    '''Check control channels during range drops'''
    
    exceeds_limit = []
    for drop in range_drops:
        gpstime = drop['gps']
        sec_chan = [
            f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ.min,s-trend',
            f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ.max,s-trend'
        ]
        start = gpstime - 100
        end = gpstime + 10

        try:
            # Grab data
            sec_data = ts.TimeSeriesDict.get(
                    sec_chan,
                    start,
                    end,
                    host=host,
                    port=port,
                    pad=float('-inf')
            )
            
            etm_dt = sec_data[f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ.min,s-trend'].dt.value
            etm_size = sec_data[f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ.min,s-trend'].value.size
            
            etm = {}
            for i in range(etm_size):
                gps = start + (etm_dt*i)
                UL_max = sec_data[f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ.max,s-trend'].value[i]
                UL_min = sec_data[f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ.min,s-trend'].value[i]
                etm[i] = make_etm_dict(gps, UL_max, UL_min)

            if ifo == 'H1':
                sec_lim = 35000
                glitch_thresh = 2 * (10 ** 4)
            elif ifo == 'L1':
                if quad == 'ETMX':
                    sec_lim = 110000
                    glitch_thresh = 5 * (10 ** 4)
                elif quad == 'ETMY':
                    sec_lim = 47000
                    glitch_thresh = 5 * (10 ** 3)
            
            for i in range(0, len(etm)):
                if (etm[i]['min'] <= -sec_lim) or (etm[i]['max'] >= sec_lim):
                    if (len(exceeds_limit) == 0) or (i == 0):
                        exceeds_limit.append(etm[i]['gps'])
                    elif (etm[i-1]['min'] > -sec_lim) or (etm[i-1]['max'] < sec_lim):
                        if etm[i]['gps'] - 2 > exceeds_limit[-1]:
                            exceeds_limit.append(etm[i]['gps'])
                    else:  
                        continue

        except RuntimeError as error:
            print(
                f'Skipping event {gpstime}: '
                f'no raw data available ({error})'
            )
            continue 


    # Time to start grabbing raw data that exceeds the limits
    interval = 1
    ETM = {}
    ETM_dt = {}
    ETM_HP_full = {}
    for event_time in exceeds_limit:
        # grab some data
        gpsStart = event_time - interval
        gpsStop = event_time + interval 

        try:
            raw_data = ts.TimeSeriesDict.get(
                [f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ'],
                gpsStart,
                gpsStop,
                host=host, 
                port=port,
            )
        except RuntimeError as error:
            print(
                f'Skipping event {event_time}: '
                f'no raw data available ({error})'
            )
            continue

        ETM[event_time] = raw_data[f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ'].value
        ETM_dt[event_time] = raw_data[f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ'].dt.value
        highpass = signal.butter(
            4,
            200,
            btype='highpass',
            output='sos',
            fs=1/ETM_dt[event_time]
        )
        highpassed_ETM = signal.sosfilt(highpass, ETM[event_time])
        ETM_HP_full[event_time] = highpassed_ETM

    # Going to start looking through high passed data
    ETM_HP = {}
    """for i in range(0, len(ETM_HP_full)):
        # Discard first 100ms of data - aka first 1638ish data values
        ETM_HP_full[exceeds_limit[i]] = ETM_HP_full[exceeds_limit[i]][1638:]
    
    # Go through and check which data sets have an etm glitch
    for i in range(0, len(ETM_HP_full)):
        glitches = np.where((ETM_HP_full[exceeds_limit[i]] <= -glitch_thresh) | (ETM_HP_full[exceeds_limit[i]] >= glitch_thresh))[0]
        if len(glitches) != 0:
            ETM_HP[exceeds_limit[i]] = ETM_HP_full[exceeds_limit[i]]"""

    for event_time, values in ETM_HP_full.items():
        discard_samples = int(0.1 / ETM_dt[event_time])

        if values.size <= discard_samples:
            print(f'Skipping short waveform for event {event_time}')
            continue

        values = values[discard_samples:]

        glitches = np.where(
            (values <= -glitch_thresh) | (values >= glitch_thresh)
        )[0]

        if glitches.size != 0:
            ETM_HP[event_time] = values
    
    # Grab gps times of the times where there are glitches in the HP
    ETM_times = [key for key, val in ETM_HP.items()]
    
    return ETM_times, ETM, ETM_dt, ETM_HP

# look at the quad master_outs, dcpds, and calib_strain
def grab_main_data(etmx_glitch):
    ''' Grabs the data for the QUAD MASTER_OUTs and DCPDs '''
    
    sec = 8
    all_data = []
    data_dict = {}
    for event_time in tqdm.tqdm(etmx_glitch, desc='Fetching and Saving glitch data', unit='event'):
        print(f'grabbing time {event_time}')
        start = event_time - sec
        end = event_time + sec
        data = ts.TimeSeriesDict.get(
            chans,
            start,
            end,
            host=host,
            port=port,
            pad=float('-inf')
        )
        # Add data to dictionary
        data_dict['timespan'] = [start, end]
        
        for i in range(len(chnnms)):
            values = data[chans[i]].value
            dt = data[chans[i]].dt.value

            data_dict[f'{chnnms[i]}_dt'] = dt
            data_dict[f'{chnnms[i]}_size'] = values.size
            data_dict[f'{chnnms[i]}'] = values
            data_dict[f'{chnnms[i]}_t'] = start + np.arange(values.size) * dt
        
        # ETM_L3
        # Get max value
        absval = abs(data_dict[f'{quad}_l3'])
        data_dict[f'{quad}_l3_absmax'] = max(absval)
        
        # CALIB_STRAIN
        # 25 Hz highpass
        hp = signal.butter(4,25, btype='highpass', output='sos', fs=1/data_dict['calib_dt'])
        calib_hp = signal.sosfilt(hp, data_dict['calib'])
        data_dict['calib_hp'] = calib_hp[1638:]
        data_dict['calib_hp_t'] = data_dict['calib_t'][1638:]
        
        all_data.append(data_dict)
        save_glitch_data(
            output_file,
            data_dict,
            event_gps=event_time,
            ifo=ifo,
            quad=quad,
        )
        data_dict = {}
        
    return all_data

def get_lockloss_times(start, end):
    '''Get the lock_state channel and record all times when it drops below 600'''

    channel = f'{ifo}:GRD-ISC_LOCK_STATE_N'

    obslock_data = ts.TimeSeries.get(
        channel,
        start=start,
        end=end,
        host=host,
        port=port,
        pad=float('-inf'),
    )

    obslock_times = obslock_data.times.value
    obslock_vals = obslock_data.value 

    # Set up Boolean masks 
    finite = np.isfinite(obslock_vals)
    locked = obslock_vals == 600
    not_locked = obslock_vals != 600
    previous_value_locked = np.roll(locked, 1)

    # Create a Boolean array that returns True/False for whether the value at a given position is the start of a lockloss
    # Must meet all three conditions to be recorded as True 
    lockloss = (
        finite 
        & not_locked
        & previous_value_locked
    )
    lockloss[0] = False 
    
    return obslock_times[lockloss].tolist()

def save_glitch_data(filename, glitch_data, event_gps, ifo):
    start, end = glitch_data['timespan']
    event_name = f'event_{event_gps:.6f}'

    with h5py.File(filename, 'a') as archive:
        events = archive.require_group('events')

        if event_name in events:
            del events[event_name]

        event = events.create_group(event_name)
        event.attrs['event_gps'] = event_gps
        event.attrs['window_start'] = start
        event.attrs['window_end'] = end
        event.attrs['ifo'] = ifo

        channels = event.create_group('channels')

        for name in chnnms:
            channel = channels.create_group(name)

            channel.create_dataset(
                'data',
                data=glitch_data[name],
                compression='gzip',
                compression_opts=4,
            )
            channel.create_dataset(
                'gps',
                data=glitch_data[f'{name}_t'],
                compression='gzip',
                compression_opts=4,
            )
            channel.attrs['name'] = chans[chnnms.index(name)] 
            channel.attrs['dt'] = glitch_data[f'{name}_dt']
            channel.attrs['size'] = glitch_data[f'{name}_size']

        highpass = channels.create_group('calib_highpass')
        highpass.create_dataset(
            'data',
            data=glitch_data['calib_hp'],
            compression='gzip',
            compression_opts=4,
        )
        highpass.create_dataset(
            'gps',
            data=glitch_data['calib_hp_t'],
            compression='gzip',
            compression_opts=4,
        )


"""
Load events like: 
with h5py.File('glitches.h5', 'r') as archive:
    event = archive['events/event_1368975618.000000']

    calib = event['channels/calib/data'][:]
    calib_gps = event['channels/calib/gps'][:]
    dcpd = event['channels/dcpd/data'][:]
    """

#######################################
# Try the same range as the lockloss tool 

parser = argparse.ArgumentParser()
parser.add_argument('--job-index', type=int, required=True)
parser.add_argument('--num-jobs', type=int, required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

num_days = 30    #292 for all of O4b
run_start = 1396796418 # O4b start time 
run_end =  1396796418 + 86400 * num_days 

if not 0 <= args.job_index < args.num_jobs:
    raise ValueError('job-index must be between 0 and num-jobs - 1')

duration = run_end - run_start

start = run_start + (duration * args.job_index) // args.num_jobs
end = run_start + (duration * (args.job_index + 1)) // args.num_jobs

output_file = args.output
output_directory = os.path.dirname(output_file)
if output_directory:
    os.makedirs(output_directory, exist_ok=True)

ifo = 'L1'#'H1'
host = 'nds.ligo-la.caltech.edu'#'nds.ligo-wa.caltech.edu'
quad = 'ETMX'
rangedropval = 140

######################################

script_start = time.perf_counter() 

chans = [
    f'{ifo}:SUS-{quad}_L1_MASTER_OUT_UL_DQ',
    f'{ifo}:SUS-{quad}_L2_MASTER_OUT_UL_DQ',
    f'{ifo}:SUS-{quad}_L3_MASTER_OUT_UL_DQ',
    f'{ifo}:OMC-DCPD_SUM_OUT_DQ',
    f'{ifo}:LSC-DARM_IN1_DQ',
    f'{ifo}:GDS-CALIB_STRAIN'
]
chnnms = [f'{quad}_l1', f'{quad}_l2', f'{quad}_l3', 'dcpd', 'darm', 'calib']
if quad == 'ETMX':
    chans.append(f'{ifo}:SUS-ETMY_L3_MASTER_OUT_UL_DQ')
    chnnms.append('etmy_l3')
elif quad == 'ETMY':
    chans.append(f'{ifo}:SUS-ETMX_L3_MASTER_OUT_UL_DQ')
    chnnms.append('etmx_l3')

print(f'On {quad}')
print(f'{gps2utc(start)} - {gps2utc(end)} UTC')
obslock_start_end = get_obs_data(start, end)
range_drops = get_range(obslock_start_end, rangedropval)
rangedrops_info = f'Range dropped below {rangedropval}Mpc {len(range_drops)} time(s)'
print(rangedrops_info)

lockloss_times = get_lockloss_times(start-1, end)          # start-1 to deal with overlap from previous segment 
lockloss_times = [
    gps for gps in lockloss_times
    if start <= gps < end
] 

with h5py.File(output_file, 'a') as archive:
    metadata = archive.require_group('metadata')

    if 'lockloss_times' in metadata:
        del metadata['lockloss_times']

    metadata.create_dataset(
        'lockloss_times',
        data = np.asarray(lockloss_times, dtype=np.float64)
    )


ETM_times, ETM, ETM_dt, ETM_HP = check_for_glitch(range_drops)
hours = stretch2totaltime(obslock_start_end)
glitches_info = f'There were {len(ETM_times)} {quad} glitches correlated ' \
                f'with range drops over the course of {hours} hours'
print(glitches_info)
all_data = grab_main_data(ETM_times)

elapsed = time.perf_counter() - script_start 
print(f'Total runtime: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)')

# Individual plots for glitch times
"""for glitch in all_data:
    start = int(glitch['timespan'][0])
    end = int(glitch['timespan'][1])
    time2utc_start = gps2utc(start)
    time2utc_end = gps2utc(end)
    thresh = 2**19
    
    plt.rcParams['figure.figsize'] = [20, 16]
    fig,ax = plt.subplots(3,3)
    axes = [ax[0,0], ax[0,1], ax[0,2], ax[1,0], ax[1,1], ax[1,2], ax[2,0]]
    
    for i in range(len(chnnms)):
        if chnnms[i] == 'calib':
            axes[i].plot(
                glitch[f'{chnnms[i]}_hp_t'] - start,
                glitch[f'{chnnms[i]}_hp']
            )
            axes[i].set_title(f'{chans[i][3:]}, highpassed at 25 Hz')
        else:
            axes[i].plot(
                glitch[f'{chnnms[i]}_t'] - start,
                glitch[f'{chnnms[i]}']
            )
            axes[i].set_title(chans[i][3:])
        
        axes[i].set_xlim(0, 3)
        axes[i].set_xlabel(f'Seconds from {glitch["timespan"][0]} [s]', fontsize=11)
        
        if chnnms[i] == f'{quad}_l1' or chnnms[i] == f'{quad}_l2' or chnnms[i] == f'{quad}_l3':
            axes[i].axhline(
                thresh,
                linestyle='--',
                color='red',
                label='Saturation threshold',
            )
            axes[i].axhline(
                -thresh,
                linestyle='--',
                color='red',
            )
        if chnnms[i].__contains__('l3'):
            axes[i].set_ylim(1.1 * -thresh, 1.1 * thresh)
        if chnnms[i] == f'{quad}_l1':
                axes[i].set_ylim(-1e5,1e5)
        if chnnms[i] == f'{quad}_l3':
            axes[i].text(0.1, thresh-0.5e5, f'abs max: {glitch[f"{quad}_l3_absmax"]}')
        if chnnms[i] == 'dcpd':
            axes[i].set_ylabel('[mA]', fontsize=11)
            if ifo == 'H1':
                axes[i].set_ylim(39.83, 40.13)  # diff of 0.3 mA
            else:
                axes[i].set_ylim(49.84, 50.14)  # diff of 0.3 mA
        if chnnms[i] == 'darm':
            if ifo == 'H1':
                axes[i].set_ylim(-8e-9,8e-9)
            else:
                axes[i].set_ylim(-10.5e-3,9.5e-3)

    fig.suptitle(f'{ifo} {quad} glitches that line up with range drops\n{time2utc_start} - {time2utc_end} UTC')
    fig.savefig(f'glitch_plots/rangedrop_{ifo}_{quad}_{start}-{end}.png')"""
