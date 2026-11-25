import gwpy.timeseries as ts
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import signal
from pathlib import Path 
import json 
from xoptimalsnr import xoptimalsnr

##################################################################

DEFAULT_DATA_FILE = 'data/O4b_glitches_test.h5'
names_file = 'correlation_results_O4b.json'

# O4c start time is 1422118818
# O4c + 1 month is 1424710818 

##################################################################

def grab_main_data(gps_times, output_file, ifo, host, port):
    ''' Grab the strain data given a set of GPS times '''
    
    channels = [f'{ifo}:SUS-ETMX_L3_MASTER_OUT_UL_DQ'] #[f'{ifo}:GDS-CALIB_STRAIN']   
    channel_names = ['calib']
    window_size = 16
    data_dict = {}
    for event_time in tqdm.tqdm(gps_times, desc='Fetching and Saving glitch data', unit='event'):
        event_time = np.floor(event_time)
        print(f'grabbing time {event_time}')
        start = event_time - (window_size/2)
        end = event_time + (window_size/2)
        data = ts.TimeSeriesDict.get(
            channels,
            start,
            end,
            host=host,
            port=port,
            pad=float('-inf')
        )
        # Add data to dictionary
        data_dict['timespan'] = [start, end]
        
        for i in range(len(channel_names)):
            channel = channels[i]
            name = channel_names[i]
        
            values = data[channel].value
            dt = data[channel].dt.value

            data_dict[f'{name}_dt'] = dt
            data_dict[f'{name}_size'] = values.size
            data_dict[f'{name}'] = values
            data_dict[f'{name}_t'] = start + np.arange(values.size) * dt
        
        # CALIB_STRAIN
        # 25 Hz highpass
        hp = signal.butter(4,200, btype='highpass', output='sos', fs=1/data_dict['calib_dt'])
        calib_hp = signal.sosfilt(hp, data_dict['calib'])

        discard_samples = int(0.1 / data_dict['calib_dt'])
        data_dict['calib_hp'] = calib_hp[discard_samples:]
        data_dict['calib_hp_t'] = data_dict['calib_t'][discard_samples:]
        
        save_glitch_data(
            output_file,
            data_dict,
            event_gps=event_time,
            ifo=ifo,
        )
        data_dict = {}

def save_glitch_data(filename, glitch_data, event_gps, ifo):
    channels = [f'{ifo}:SUS-ETMX_L3_MASTER_OUT_UL_DQ'] #[f'{ifo}:GDS-CALIB_STRAIN']   
    channel_names = ['calib']

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

        channel_group = event.create_group('channels')

        for name in channel_names:
            channel = channel_group.create_group(name)

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
            channel.attrs['name'] = channels[channel_names.index(name)] 
            channel.attrs['dt'] = glitch_data[f'{name}_dt']
            channel.attrs['size'] = glitch_data[f'{name}_size']

        highpass = channel_group.create_group('calib_highpass')
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


def get_event_names(filename='data/O4_glitches.h5'):
    with h5py.File(filename, 'r') as archive:
        return list(archive['events'].keys())

def _read_event(archive, event_name):
    event = archive['events'][event_name] 

    event_data = {
        'name' : event_name,
        'event_gps' : event.attrs['event_gps'],
        'window_start' : event.attrs['window_start'],
        'window_end' : event.attrs['window_end'],
        'ifo' : event.attrs['ifo'],
        'channels' : {},
    }

    for channel_name, channel in event['channels'].items():
        event_data['channels'][channel_name] = {
            'values' : channel['data'][:],
            'gps' : channel['gps'][:],
        }

        if 'dt' in channel.attrs:
            event_data['channels'][channel_name]['dt'] = channel.attrs['dt']

    return event_data 

def iter_events(filename=DEFAULT_DATA_FILE):
    with h5py.File(filename, 'r') as archive:
        for event_name in archive['events']:
            yield _read_event(archive, event_name) 

def load_event(event_name, filename=DEFAULT_DATA_FILE):
    with h5py.File(filename, 'r') as archive:
        return _read_event(archive, event_name) 

'''
# This is the old load function, which was too memory-intensive 
def load_event(event_name, filename='data/glitches.h5'):

    event_data = {
        'name': event_name,
        'channels': {},
    }
        
    with h5py.File(filename, 'r') as archive: 
        event = archive['events'][event_name]

        event_data['event_gps'] = event.attrs['event_gps']
        event_data['window_start'] = event.attrs['window_start']
        event_data['window_end'] = event.attrs['window_end']
        event_data['ifo'] = event.attrs['ifo']
        event_data['quad'] = event.attrs['quad']

        for channel_name, channel in event['channels'].items():
            event_data['channels'][channel_name] = {
                'values': channel['data'][:],
                'gps': channel['gps'][:],
            }

            if 'dt' in channel.attrs:
                event_data['channels'][channel_name]['dt'] = (
                    channel.attrs['dt']
                )

    return event_data
'''

def psd(timeseries):
    timeseries = ts.TimeSeries(timeseries)    # convert to gwpy TimeSeries object 
    psd = timeseries.psd(fftlength=4, overlap=None, window='hann', method = 'welch')
    return psd 

def plot_events(events):
    
    channel_names = ['calib', 'calib_highpass']

    Path('template_plots').mkdir(exist_ok=True)

    for event in events:
        event_name = event['name']
        event_gps = event['event_gps']
        ifo = event['ifo']

        fig, axes = plt.subplots(
            1, 
            len(channel_names),
            figsize=(24, 6),
            squeeze=False,
        )

        axes = axes[0] 

        for column, channel_name in enumerate(channel_names):
            axis = axes[column]

            channel = event['channels'].get(channel_name)

            if channel is None:
                axis.set_visible(False)
                continue

            relative_time = channel['gps'] - event_gps

            y_lim = [
                (-4e5,4e5),
                (-4e4,4e4),
            ]

            # Zoomed window around event trigger 
            axis.plot(relative_time, channel['values'], linewidth=0.6)
            axis.set_ylim(*y_lim[column])     # * tells python to unpack tuple 
            axis.set_xlim(-8,8)
            axis.set_ylabel(channel_name)
            axis.set_title('Zoomed 2s window after trigger time')
            axis.grid(True)

        fig.suptitle(event_name)
        fig.tight_layout()
        fig.savefig(f'ETM_plots/{event_name}.png', dpi=180)
        plt.close(fig)


def whiten(data):
    data = ts.TimeSeries(data) # Needs to be a timeseries to use whiten method 
    white_data = data.whiten(4, 2, fduration=4, method='median')
    return white_data

def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

def calculate_event_metrics(event, channel_name='calib'):
    channel = event['channels'].get(channel_name)
    if channel is None:
        raise KeyError(f"Event {event['name']} has no '{channel_name}' channel")

    waveform = np.asarray(channel['values'], dtype=np.float64)
    gps = np.asarray(channel['gps'], dtype=np.float64)

    if waveform.size < 2:
        raise ValueError("Too few waveform samples")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Waveform contains non-finite values")

    dt = np.median(np.diff(gps))
    fs = 1.0 / dt

    # Analyze a 0.4-second window around the largest absolute sample.
    peak_idx = np.argmax(np.abs(waveform))
    half_window = int(round(0.2 * fs))
    start = max(0, peak_idx - half_window // 2)
    end = min(waveform.size, peak_idx + half_window)

    waveform_window = waveform[start:end]
    if waveform_window.size < 2:
        raise ValueError("Analysis window has too few samples")

    # Calculate the noise PSD from the complete event window.
    timeseries = ts.TimeSeries(waveform, sample_rate=fs)
    psd_result = timeseries.psd(
        fftlength=4,
        overlap=2,
        method='median',
        window='hann',
    )

    metrics = xoptimalsnr(
        h=waveform_window[:, np.newaxis],
        t0=0.0,
        fs=fs,
        S=psd_result.value,
        F0=psd_result.frequencies.value,
        dF=None,
        Fmin=25,
        Fmax=fs / 2,
    )

    (
        snr,
        h_rss,
        h_peak,
        fchar,
        bandwidth,
        tchar,
        duration,
        tchar_weighted,
        duration_weighted,
        tfbox,
        h_weighted,
    ) = metrics

    return {
        'event_name': event['name'],
        'event_gps': event['event_gps'],
        'snr': snr,
        'h_rss': h_rss,
        'h_peak': h_peak,
        'fchar': fchar,
        'bandwidth': bandwidth,
        'tchar': tchar,
        'duration': duration,
        'tchar_weighted': tchar_weighted,
        'duration_weighted': duration_weighted,
    }

##################################################################

if __name__ == '__main__':

    with open(names_file, 'r') as file:
        glitch_names = set(json.load(file)['glitch_names'])

    for event in iter_events():
        if event['name'] not in glitch_names:
            continue

        idx = 0
        plot_events([event])

        idx += 1 
        print(idx)

        if idx >=10: 
            break
        