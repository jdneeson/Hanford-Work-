import json

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

from load_data import iter_events, load_event
from tqdm import tqdm 
from pathlib import Path 
import argparse 

#########################################

'''
- Split each event into sliding windows 
- For each A, calculate max correlation with all Bs; can be vectorised with matrices 
- Take max of those and store it. Loop over this process to do it for all As. 
- Compare the overall maximum to the threshold 
'''


DATA_FILE = 'data/O4b_glitches_test.h5'
threshold = 0.82   # ~0.5 for 0 false negatives, ~0.8 for no false positives

# Old template bank when using strain channel 
selected_names = {
    'event_1396832853.000000',
    #'event_1396841611.000000',  
    #'event_1396905742.000000',
    'event_1396943944.000000',   # Necessary for threshold of 0.5
    'event_1396960782.000000',
    #'event_1396963806.000000',
    #'event_1397033913.000000',
    #'event_1397052394.000000',
    #'event_1397067248.000000',
    'event_1397089211.000000',   # Necessary for threshold of 0.5
    'event_1397123118.000000',
    #'event_1397138111.000000',
    #'event_1397152128.000000',
    #'event_1397216198.000000',
    'event_1397419479.000000',
    'event_1397556350.000000',   # Necessary for threshold of 0.5 # add back in 
    #'event_1397669097.000000',
    #'event_1397706176.000000',
    'event_1397779267.000000',
    #'event_1397896895.000000',
    'event_1397903973.000000',
    #'event_1398118929.000000',
    'event_1398310825.000000',   # Necessary for threshold of 0.5
    #'event_1398356224.000000',
    #'event_1398458160.000000',
    'event_1398450883.000000',   # Necessary for threshold of 0.5
    #'event_1398581067.000000',
    #'event_1398587917.000000',
    #'event_1398745963.000000',
    #'event_1398561301.000000',                   
    #'event_1398581067.000000',
    #'event_1398587917.000000',
    #'event_1398748571.000000',
    'event_1398984684.000000',   # Necessary for threshold of 0.5  # add back in
    'event_1399005063.000000',   # add back in 
    #'event_1399022424.000000',
    #'event_1399085885.000000',
    #'event_1399210049.000000',
    #'event_1399255111.000000',
    #'event_1399194406.000000',
    #'event_1398779196.000000',
    #'event_1398089719.000000',
    #'event_1397844188.000000',
    #'event_1397679085.000000',
    #'event_1398659390.000000',
    #'event_1398932686.000000',
    #'event_1398974450.000000',
}

#########################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()

def whiten(data, dt):
    data = TimeSeries(data, sample_rate = 1/dt) # Needs to be a timeseries to use whiten method 
    white_data = data.whiten(4, 2, fduration=4, method='median', window='hann')
    return white_data

def prepare_event(event):
    channel = event['channels']['calib_highpass']
    dt = event['channels']['calib']['dt']
    strain = whiten(channel['values'], dt)

    return {
        'name' : event['name'],
        'strain' : strain,
        'dt' : dt,
    }

def sliding_window(data, signal_duration, segment_length, overlap_length, sample_rate, ):
    
    """
    Splits a signal into overlapping windows 

    Parameters
    ----------
    data: array-like
        The signal that you want to split into sections 
    
    signal_duration: float
        The duration of the entire signal, in seconds.

    segment_length: float
        The desired length of each window, in seconds. 

    overlap_length: float
        The desired overlap between consecutive windows, in seconds. 

    sample_rate: float 
        The sample rate of the signal, in Hz. 

    """
    data_size = int(signal_duration*sample_rate)                         
    segment_size = int(segment_length*sample_rate)                          
    overlap_size = int(overlap_length*sample_rate)  
    step = int(segment_size - overlap_size)                                    
    num_windows = int((data_size-segment_size)/(segment_size-overlap_size)) 

    #create the indices to split the signal into overlapping chunks 
    start_indices = []                                 
    end_indices = [] 
    for n in range(0, num_windows+1, 1): 
        start_index = 0 + n*step 
        start_indices.append(start_index) 
        end_index = segment_size + n*step 
        end_indices.append(end_index) 
    start_indices = np.array(start_indices)
    end_indices = np.array(end_indices)
    #print(start_indices)
    #print(end_indices)

    #split the signal into overlapping chunks 
    chunks = []
    for i in range(len(start_indices)): 
        chunk = data[start_indices[i]:end_indices[i]]
        chunks.append(chunk)

    return chunks, segment_size 

def corr(data1, data2, dt, restrict_to_trigger=True):
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    restriction_start = 0
    restriction_end = len(data1)

    if restrict_to_trigger:
        sample_rate = 1.0 / dt
        restriction_start = int(round(8 * sample_rate))
        restriction_end = int(round(10 * sample_rate))

        data1 = data1[restriction_start:restriction_end]
        data2 = data2[restriction_start:restriction_end]

    data1_windows, segment_size = sliding_window(
        data1, 2, 1.5, 1.48, 1 / dt
    )
    data2_windows, segment_size = sliding_window(
        data2, 2, 1.5, 1.48, 1 / dt
    )

    data1_windows = np.asarray(data1_windows, dtype=np.float64)
    data2_windows = np.asarray(data2_windows, dtype=np.float64)

    data1_windows /= np.linalg.norm(data1_windows, axis=1, keepdims=True)
    data2_windows /= np.linalg.norm(data2_windows, axis=1, keepdims=True)

    fft1 = np.fft.rfft(data1_windows, axis=1, n=segment_size)
    fft2 = np.fft.rfft(data2_windows, axis=1, n=segment_size)

    correlations = np.fft.irfft(
        fft1[:, np.newaxis, :] * np.conjugate(fft2[np.newaxis, :, :]),
        axis=2,
        n=segment_size,
    )

    event_window_index, template_window_index, lag_index = np.unravel_index(
        np.argmax(np.abs(correlations)),
        correlations.shape,
    )

    step = segment_size - int(round(1.48 * 1 / dt))

    event_window_start = restriction_start + event_window_index * step
    template_window_start = restriction_start + template_window_index * step

    window_indices = {
        "event": {
            "start": int(event_window_start),
            "end": int(event_window_start + segment_size),
        },
        "template": {
            "start": int(template_window_start),
            "end": int(template_window_start + segment_size),
        },
        "lag": int(lag_index),
    }

    return float(np.max(np.abs(correlations))), window_indices

####################################################################

def main(args):
    selected_events = {}

    # Load selected template events
    for event in iter_events(args.templates):
        if event["name"] in selected_names:
            selected_events[event["name"]] = prepare_event(event)

    missing_names = selected_names - selected_events.keys()
    if missing_names:
        raise ValueError(f"Selected events not found: {missing_names}")

    matches = {}

    # Search every event
    for event in iter_events(args.data):
        event_name = event["name"]

        event_data = (
            selected_events[event_name]
            if event_name in selected_events
            else prepare_event(event)
        )

        best_match = None
        best_correlation = -np.inf

        for selected_name, selected_event in selected_events.items():
            correlation, window_indices = corr(
                event_data["strain"],
                selected_event["strain"],
                event_data["dt"],
            )

            if correlation > best_correlation:
                best_correlation = correlation
                best_match = {
                    "event_name": event_name,
                    "template_name": selected_name,
                    "correlation": float(correlation),
                    "event_window": window_indices["event"],
                    "template_window": window_indices["template"],
                    "lag": window_indices["lag"],
                }

        if best_correlation >= threshold:
            matches[event_name] = best_match

    results = {
        "templates": sorted(selected_names),
        "glitch_names": sorted(matches),
        "matches": [matches[name] for name in sorted(matches)],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    
'''
    true_positives = selected_names & events_that_pass
    false_negatives = selected_names - events_that_pass
    false_positives = events_that_pass - selected_names
    true_negatives = all_names - selected_names - events_that_pass

    #print(f'False Positives: {false_positives}') 

    #print(f'False Negatives: {false_negatives}') 

    #print(f'True Positives: {true_positives}')

    confusion = np.array([
        [len(true_negatives), len(false_positives)],
        [len(false_negatives), len(true_positives)],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Ignored', 'Found'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Noise', 'Glitch'])

    ax.set_xlabel('Classifier result')
    ax.set_ylabel('Manual classification')
    ax.set_title(f'Confusion Matrix, correlation threshold = {threshold}')

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                confusion[row, column],
                ha='center',
                va='center',
                color='white' if confusion[row, column] > confusion.max() / 2 else 'black',
            )

    fig.colorbar(image, ax=ax, label='Number of events')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
'''

if __name__ == '__main__':
    main(parse_args())

'''
Simple plot of aligned glitches to look for similar morphology 
- Find the peak of each event 
- Take 1/10th of a second around the peak 
- If negative multiply by -1
- Plot on top of each other (with thin lines)
- Average and compute correlations with average to see how much variation there is 
- Try the same set over O4 data; if it doesn't take very long try with the full original set to see if there's much difference

event_windows = {}

for event_name, event in selected_events.items():
    strain = np.asarray(event['strain'], dtype=float).copy()
    dt = event['dt']

    peak_idx = np.argmax(np.abs(strain))
    interval = int(round(0.1/dt))          #Find number of samples for 0.1s

    if strain[peak_idx] < 0:
        strain *= -1
    
    strain /= np.max(np.abs(strain))

    start_idx = peak_idx - interval
    end_idx = peak_idx + interval 
    window = strain[start_idx:end_idx] 

    event_windows[event_name] = window

fig, ax = plt.subplots(figsize=(12,6))

for name, window in event_windows.items():
    dt = selected_events[name]['dt']
    time = dt * (np.arange(len(window)) - len(window)//2) 

    ax.plot(time, window, linewidth = 0.5, alpha = 0.8)

# Calculate and plot average 
window_array = np.stack(list(event_windows.values()))   # Convert to 2D array, rows = events, columns = time samples 
average_glitch = np.mean(window_array, axis=0) 
average_time = dt * (np.arange(len(average_glitch)) - len(average_glitch)//2) 

for name, window in event_windows.items():
    dt = selected_events[name]['dt']
    correlation = corr(window, average_glitch, dt, restrict_to_trigger=False)
    peak_correlation = np.max(abs(correlation))
    print(f'Correlation for {name} : {peak_correlation}')

#ax.plot(average_time, average_glitch, color='black', linewidth=1, label ='average')
#ax.legend()
ax.set_xlabel('Time relative to peak (s)')
ax.set_ylabel('Whitened strain')
ax.set_title('Aligned Glitches')
fig.tight_layout()
plt.savefig('Aligned_Events.png') 
'''