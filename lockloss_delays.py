import json

import h5py
import matplotlib.pyplot as plt
import numpy as np

with open('locklosses.json', 'r') as file: 
    lockloss_data = json.load(file)

lockloss_times = np.array(
    [event['gps'] for event in lockloss_data['events']], 
    dtype=np.float64,
)

lockloss_times.sort()
print(len(lockloss_times))

with open('correlation_results_O4b.json', 'r') as file:
    correlated_glitches = json.load(file)

correlated_glitch_names = set(correlated_glitches['glitch_names']) 

with h5py.File('data/hveto_glitches.h5', 'r') as archive:
    selected_events = (
        archive['events'][name] 
        for name in correlated_glitch_names 
        if name in archive['events']
    )
    glitch_times = np.array(
        [event.attrs['event_gps'] for event in selected_events],
        dtype=np.float64,
    ) 

next_lockloss_indices = np.searchsorted(lockloss_times, glitch_times, side='left') 
has_next_lockloss = next_lockloss_indices < len(lockloss_times)        # Mask for glitches that occur after the last lockloss

delay_until_lockloss = np.full(glitch_times.shape, np.nan, dtype=np.float64)  # Initialise array with NaNs so that values without a next_lockloss inherit that value
delay_until_lockloss[has_next_lockloss] = lockloss_times[next_lockloss_indices[has_next_lockloss]] - glitch_times[has_next_lockloss]

valid_delays = delay_until_lockloss[np.isfinite(delay_until_lockloss)]

bins = np.logspace(np.log10(valid_delays.min()), np.log10(valid_delays.max()), 100)

plt.hist(valid_delays, bins=bins)
plt.xscale('log')
plt.xlabel('Time from glitch to next lock loss [s]')
plt.ylabel('Number of glitches')
plt.title('Delay from glitch to next lock loss')
plt.savefig('lockloss_delays.png')