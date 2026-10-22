import json

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

from load_data import iter_events, load_event

#########################################

DATA_FILE = 'data/hveto_glitches.h5'
threshold = 0.8   # ~0.5 for 0 false negatives, ~0.8 for no false positives

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
    #'event_1399005063.000000',   # add back in 
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

def corr(data1, data2, dt, restrict_to_trigger=True):
    '''
    Correlation in wraparound lag order.

    ans[0]     = zero lag
    ans[1:]    = increasingly positive lags
    ans[n//2:] = increasingly negative lags, wrapping around

    The input length must be the same power of two.
    '''

    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)

    if restrict_to_trigger:
    # Restrict to 2s either side of trigger 
        start_idx = int(8 * 1/dt)      # Currently hardcoded for convenience 
        end_idx = int(10 * 1/dt) 
        data1 = data1[start_idx:end_idx]
        data2 = data2[start_idx:end_idx]
    data1 = data1 / np.linalg.norm(data1)
    data2 = data2 / np.linalg.norm(data2) 

    if data1.shape != data2.shape:
        raise ValueError('data1 and data2 must have the same shape')

    n = data1.size

    #if n == 0 or n & (n - 1):
    #    raise ValueError('input length must be a power of two')

    fftlength = n 

    fft1 = np.fft.rfft(data1)
    fft2 = np.fft.rfft(data2)

    # Multiplication by the complex conjugate gives the correlation.
    ans = np.fft.irfft(fft1 * np.conjugate(fft2), n=fftlength)

    return ans 

####################################################################

def main():

    selected_events = {}

    # Pre-load only the selected events that will be compared against all other events 
    for event in iter_events(DATA_FILE):
        if event['name'] in selected_names:
            selected_events[event['name']] = prepare_event(event) 

    missing_names = selected_names - selected_events.keys()
    if missing_names:
        raise ValueError(f'Selected events not found: {missing_names}') 

    events_that_pass = set()
    all_names = set() 

    # Load and analyse the events that will only be used once one at a time using the iter_events() generator
    for event in iter_events(DATA_FILE):
        event_name = event['name'] 
        all_names.add(event_name) 
    
        if event_name in selected_events:
            event_data = selected_events[event_name]
        else:
            event_data = prepare_event(event) 

        for selected_name, selected_event in selected_events.items():
            if event_name == selected_name:
                continue 

            correlation = corr(
                event_data['strain'],
                selected_event['strain'],
                event_data['dt'],
            )

            peak_correlation = np.max(abs(correlation))

            if peak_correlation >= threshold: 
                events_that_pass.add(event_name)
                break      # Only care if it meets the threshold once and want to avoid duplicates

    results = {
        'data_file' : DATA_FILE,
        'threshold' : threshold,
        'templates' : sorted(selected_names), 
        'glitch_names' : sorted(events_that_pass),
        }
    
    with open('correlation_results_O4b.json', 'w') as file:
        json.dump(results, file, indent=2)

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
    fig.savefig('confusion_matrix_O4b.png', dpi=150)

def test():
    test_event = load_event(
        'event_1398450883.000000',#'event_1398450883.292968',
        filename=DATA_FILE,
    )
    test_event2 = load_event(
        'event_1397903973.000000',#'event_1397903973.529297',
        filename=DATA_FILE,
    )
    event = prepare_event(test_event)
    event2 = prepare_event(test_event2)
    correlation = corr(event['strain'], event2['strain'], event['dt'])
    print(np.max(np.abs(correlation)))
    

if __name__ == '__main__':
    main()

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