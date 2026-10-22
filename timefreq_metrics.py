from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

from xoptimalsnr import xoptimalsnr

###########################################################
# Load data and calculate metrics using xoptimalsnr
###########################################################

input_file = 'data/test_glitches_new.h5'
channel_name = 'calib_highpass' 

results = [] 

with h5py.File(input_file, 'r') as archive:
    for event_name, event in archive['events'].items():
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

        timeseries = TimeSeries(
            waveform, 
            sample_rate = fs
        )

        psd = timeseries.psd(
            fftlength=4,
            overlap=2,
            method='median',
            window='hann',
        )

        # xoptimalsnr expects shape (samples, polarisations)
        h = waveform[:, np.newaxis]

        metrics = xoptimalsnr(
            h=h,
            t0=0.0,    # This is relative time for the trigger
            fs=fs,
            S=psd.value,
            F0=psd.frequencies.value,
            dF=None,
            Fmin=25,   # Data is highpassed at 20Hz
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

plot_directory = Path('plots/timefreq_metrics')
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
            bins=30,
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