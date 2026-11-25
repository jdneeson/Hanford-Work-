from pathlib import Path
from tqdm import tqdm 

import h5py
import json 
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

from plot_data import plot_events 

from xoptimalsnr import xoptimalsnr

###########################################################
# Load data and calculate metrics using xoptimalsnr
###########################################################

input_file = 'data/O4b_glitches_test.h5'
channel_name = 'calib' 

results = [] 

with open('correlation_results_test.json', 'r') as file:
    correlated_glitches = json.load(file)
    correlated_glitch_names = set(correlated_glitches['glitch_names']) | set(correlated_glitches['old_glitch_names'])

with h5py.File(input_file, 'r') as archive:
    events = archive['events']

    selected_names = [
        name for name in correlated_glitch_names
        if name in events
    ]

    selected_events = (
        (name, archive['events'][name])
        for name in selected_names
    )

    for event in selected_events:
        for event_name, event in tqdm(
                selected_events,
                total=len(selected_names),
                desc='Calculating metrics',
                unit='event',
            ):
            channel = event['channels'][channel_name] 

            waveform = np.asarray(channel['data'][:], dtype=np.float64)
            gps = np.asarray(channel['gps'][:], dtype=np.float64)

            if waveform.size < 2:
                print(f'Skipping {event_name}: too few samples')
                continue

            if not np.all(np.isfinite(waveform)):
                print(f'Skipping {event_name}: non-finite waveform values')
                continue

            dt = np.median(np.diff(gps))
            fs = 1/dt 

            peak_idx = np.argmax(np.abs(waveform))

            half_window = int(round(0.2*fs))
            start = max(0, peak_idx - int(half_window/2))
            end = min(waveform.size, peak_idx + half_window)

            waveform_window = waveform[start:end]
            t0 = (peak_idx - start) / fs 

            timeseries = TimeSeries(
                waveform, 
                sample_rate = fs
            )

            # PSD still calculated from full data, is this correct? 
            psd = timeseries.psd(
                fftlength=4,
                overlap=2,
                method='median',
                window='hann',
            )

            # xoptimalsnr expects shape (samples, polarisations)
            h = waveform_window[:, np.newaxis]

            metrics = xoptimalsnr(
                h=h,
                t0=t0,    # This is relative time for the trigger
                fs=fs,
                S=psd.value,
                F0=psd.frequencies.value,
                dF=None,
                Fmin=25,   # Data is highpassed at 25Hz
                Fmax=fs/2, # Nyquist frequency 
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
                tfbox,       # Currently omitted, not scalar so would need to be stored differently 
                h_weighted,  # Not stored to keep memory costs low, will need to modify if it would be useful later 
            ) = metrics

            results.append({
                'event_name': event_name,
                'event_gps': event.attrs['event_gps'],
                'snr': snr,
                'h_rss': h_rss,
                'h_peak': h_peak,
                'fchar': fchar,
                'bandwidth': bandwidth,
                'tchar': tchar,
                'duration': duration,
                'tchar_weighted': tchar_weighted,
                'duration_weighted': duration_weighted,
            })

print(f'Calculated metrics for {len(results)} events')

########################################
# Save metrics for later use
########################################

metric_fields = [
    'event_gps',
    'snr',
    'h_rss',
    'h_peak',
    'fchar',
    'bandwidth',
    'tchar',
    'duration',
    'tchar_weighted',
    'duration_weighted',
]

event_names = np.asarray(
    [result['event_name'] for result in results],
    dtype=h5py.string_dtype(encoding='utf-8'),
)

metric_arrays = {
    field: np.asarray(
        [result[field] for result in results],
        dtype=np.float64,
    )
    for field in metric_fields
}

with h5py.File(input_file, 'r+') as archive:
    metrics = archive.require_group('metrics')

    if 'xoptimalsnr' in metrics:
        del metrics['xoptimalsnr']

    xoptimalsnr_group = metrics.create_group('xoptimalsnr')

    xoptimalsnr_group.create_dataset(
        'event_name',
        data=event_names,
    )

    for field, values in metric_arrays.items():
        xoptimalsnr_group.create_dataset(
            field,
            data=values,
        )

#####################################################################
# Plot results 
#####################################################################

plot_directory = Path('plots/timefreq_metrics/test')
plot_directory.mkdir(parents=True, exist_ok=True)

plot_fields = [
    'snr',
    'h_rss',
    'h_peak',
    'fchar',
    'bandwidth',
    'tchar',
    'duration',
    'tchar_weighted',
    'duration_weighted',
]

with h5py.File(input_file, 'r') as archive:
    xoptimalsnr_group = archive['metrics/xoptimalsnr']

    for field in plot_fields:
        values = np.asarray(xoptimalsnr_group[field][:], dtype=np.float64)
        values = values[np.isfinite(values)]

        if values.size == 0:
            print(f'Skipping {field}: no finite values')
            continue

        mean = np.mean(values)
        std = np.std(values)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            values,
            bins=100,
            color='steelblue',
            alpha=0.75,
            edgecolor='black',
            linewidth=0.5,
        )

        ax.axvline(
            mean,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'mean = {mean:.3g}',
        )

        ax.axvspan(
            mean - std,
            mean + std,
            color='black',
            alpha=0.10,
            label=f'±1 std = {std:.3g}',
        )

        ax.set_xlabel(field)
        ax.set_ylabel('Number of events')
        ax.set_title(
            f'{field} distribution\n'
            f'mean = {mean:.3g}, std = {std:.3g}, '
            f'N = {values.size}'
        )
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.25)

        fig.tight_layout()
        fig.savefig(plot_directory / f'{field}.png', dpi=150)
        plt.close(fig)