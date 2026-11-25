import gwpy.timeseries as ts
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import signal
from pathlib import Path 

##################################################################

DEFAULT_DATA_FILE = 'data/O4b_glitches-ETM-SUS.h5'

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
        hp = signal.butter(4,25, btype='highpass', output='sos', fs=1/data_dict['calib_dt'])
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

'''
def iter_events(filename=DEFAULT_DATA_FILE):
    with h5py.File(filename, 'r') as archive:
        for event_name in archive['events']:
            yield _read_event(archive, event_name) 
'''

def iter_events(filename=DEFAULT_DATA_FILE):
    source = Path(filename)
    files = sorted(source.glob("*.h5")) if source.is_dir() else [source]
    seen = set()

    for path in files:
        with h5py.File(path, "r") as archive:
            if "events" not in archive:
                continue

            for event_name in archive["events"]:
                # If an event is in more than one file, use its first match.
                if event_name in seen:
                    continue
                seen.add(event_name)

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


def whiten(data):
    data = ts.TimeSeries(data) # Needs to be a timeseries to use whiten method 
    white_data = data.whiten(4, 2, fduration=4, method='median')
    return white_data