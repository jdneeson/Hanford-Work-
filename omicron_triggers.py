from urllib.parse import urlparse
import numpy as np
from gwtrigfind import find_trigger_files
from gwpy.table import EventTable
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# event_file = "refined_gps_events.txt"

channel = "H1:GDS-CALIB_STRAIN"
etg = "Omicron"

# seconds around lock loss to search for triggers
window_start = 2.0
window_end = 2.0
snr_threshold = 50 

#with open(event_file, "r", encoding='utf-8') as f:
#    gps_times = f.read()
# print(f"Loaded {len(gps_times)} ETM_GLITCH events")

# df = pd.read_csv(event_file, sep=r"\s+", dtype={'gps_time': 'Int64'})

#first_column = df.iloc[:, 0]

first_column = [1422118818, 1447459218]#1424710818]

start = first_column[0]
stop = first_column[-1]
step = 1

full_gps_times = list(range(start, stop + step, step))

print(full_gps_times)
results = []

for i, gps in enumerate(full_gps_times):

    print(f"{i+1}/{len(full_gps_times)} Refined GPS {gps}")
    
    try:
        files = find_trigger_files(channel, etg, int(gps-window_start), int(gps+window_end))
        #print(files)
    except Exception as e:
        print(e)
        continue

    for fname in files:
        path = urlparse(fname).path
        triggers = EventTable.read(path, path='triggers')
        nearby = triggers[
            (triggers["time"] >= gps-window_start) &
            (triggers["time"] < gps+window_end) & 
            (triggers["snr"] >= snr_threshold)
        ]

        if len(nearby) == 0: 
            continue

        #best = nearby[np.argmax(nearby["snr"])]
        # try sorting by proximity
        dt = np.abs(nearby["time"] - gps)
        nearby = nearby[np.argsort(dt)]
        candidates = nearby[:5]
        best = candidates[np.argmax(candidates["snr"])]
        #print(best.keys())

        results.append({
            "gps_time": (f'{gps:.2f}'),
            "trigger_time": int(best["time"]),
            "frequency": float(best["frequency"]),
            "tstart": best['tstart'],
            "tend": best['tend'],
            "fstart": best['fstart'],
            "fend": best['fend'],
            "snr": float(best["snr"]),
            "q": float(best["q"]),
            "amplitude": float(best["amplitude"]),
            "phase":  float(best["phase"]),
            })

        #plot = triggers.scatter("time", "frequency", color="snr")
        #ax = plot.gca()
        #ax.set_xlim(int(gps-window_start), int(gps+window_end))
        #ax.set_yscale('log')
        #ax.set_ylim(10, 2000)
        #ax.set_ylabel('Frequency [Hz]')
        #plot.colorbar(cmap='YlGnBu', clim=[3, 100], label='Signal-to-noise ratio (SNR)')
        #plot.show()
        #plot.savefig('./time_vs_freq.png')
        #plot.close()

df = pd.DataFrame(results)
print(f"Found {len(df)} Omicron triggers")

df.to_csv("ETM_GLITCH_omicron_triggers_sel.csv", index=False,)
print(df.head())
