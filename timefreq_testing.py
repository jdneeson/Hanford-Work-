from pathlib import Path
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import welch
from tqdm import tqdm

from xoptimalsnr import xoptimalsnr


INPUT_FILE = 'data/O4b_glitches_test.h5'
NAMES_FILE = 'correlation_results_O4b.json'
CHANNEL_NAME = 'calib_highpass'
METRIC_DIR = Path('metric_results')
PLOT_DIR = Path('metric_plots')

METRIC_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def safe_filename(name):
    return ''.join(
        character if character.isalnum() or character in '-_.'
        else '_'
        for character in name
    )


def plot_metric_event(
    event_name,
    event_gps,
    start_gps,
    waveform_2s,
    gps_2s,
    waveform_window,
    gps_window,
    fs,
    result,
):
    relative_time = np.asarray(gps_2s) - start_gps
    window_relative = np.asarray(gps_window) - start_gps
    waveform_window = np.asarray(waveform_window)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=False,
        constrained_layout=True,
    )
    time_axis, spectrum_axis = axes

    time_axis.plot(
        relative_time,
        waveform_2s,
        linewidth=0.6,
        label='2-second post-trigger waveform',
    )
    time_axis.plot(
        window_relative,
        waveform_window,
        linewidth=1.2,
        color='crimson',
        label='correlation window',
    )

    time_axis.axvspan(
        window_relative[0],
        window_relative[-1],
        color='red',
        alpha=0.15,
    )

    time_axis.axvline(
        window_relative[0] + result['tchar'],
        color='black',
        linestyle='--',
        label='Characteristic time',
    )

    time_axis.set_xlim(8, 10)
    time_axis.set_xlabel('Seconds (Trigger at 8s)')
    time_axis.set_ylabel('Amplitude')
    time_axis.legend(frameon=False)
    time_axis.grid(alpha=0.25)

    frequencies, spectrum = welch(
        waveform_window,
        fs=fs,
        window='hann',
        nperseg=min(4096, waveform_window.size),
        detrend='constant',
    )

    band_mask = (
        (frequencies >= 25)
        & (frequencies <= fs / 2)
    )

    spectrum_axis.semilogy(
        frequencies[band_mask],
        spectrum[band_mask],
        linewidth=0.8,
        color='darkorange',
    )
    spectrum_axis.axvline(
        result['fchar'],
        color='black',
        linestyle='--',
        label='Characteristic frequency',
    )

    spectrum_axis.set_xlim(25, fs / 2)
    spectrum_axis.set_xlabel('Frequency (Hz)')
    spectrum_axis.set_ylabel('PSD')
    spectrum_axis.legend(frameon=False)
    spectrum_axis.grid(alpha=0.25)

    fig.suptitle(event_name)
    fig.savefig(PLOT_DIR / f'{safe_filename(event_name)}.png', dpi=180)
    plt.close(fig)


with open(NAMES_FILE, 'r', encoding='utf-8') as file:
    correlated_glitches = json.load(file)

selected_names = set(correlated_glitches['glitch_names'])

# Look up the matched window for each event.
matches_by_event = {
    match['event_name']: match
    for match in correlated_glitches.get('matches', [])
}

processed = 0

with h5py.File(INPUT_FILE, 'r') as archive:
    events = archive['events']

    names_to_process = sorted(
        name for name in selected_names
        if name in events
    )

    for event_name in tqdm(
        names_to_process,
        desc='Calculating metrics and plotting',
        unit='event',
    ):
        event = events[event_name]

        if CHANNEL_NAME not in event['channels']:
            print(f'Skipping {event_name}: channel {CHANNEL_NAME} is missing')
            continue

        match = matches_by_event.get(event_name)
        if match is None:
            print(f'Skipping {event_name}: no correlation match found')
            continue

        channel = event['channels'][CHANNEL_NAME]
        waveform = np.asarray(channel['data'][:], dtype=np.float64)
        gps = np.asarray(channel['gps'][:], dtype=np.float64)

        event_gps = json_value(event.attrs['event_gps'])

        if waveform.size != gps.size:
            print(f'Skipping {event_name}: waveform and GPS lengths differ')
            continue

        if not np.all(np.isfinite(waveform)):
            print(f'Skipping {event_name}: non-finite waveform values')
            continue

        dt = np.median(np.diff(gps))
        fs = 1.0 / dt

        start_idx = int(round(8.0 * fs))
        end_idx = int(round(10.0 * fs))

        waveform_2s = waveform[start_idx:end_idx]
        gps_2s = gps[start_idx:end_idx]

        if waveform_2s.size < 2:
            print(f'Skipping {event_name}: too few post-trigger samples')
            continue

        # Use the .5-second window saved by the correlation search.
        matched_window = match['event_window']
        start = int(matched_window['start'])
        end = int(matched_window['end'])

        if start < 0 or end > waveform.size or end <= start:
            print(
                f'Skipping {event_name}: invalid matched window '
                f'({start}, {end})'
            )
            continue

        waveform_window = waveform[start:end]
        gps_window = gps[start:end]

        if waveform_window.size < 2:
            print(f'Skipping {event_name}: matched window is too small')
            continue

        dt = np.median(np.diff(gps_window))
        fs = 1.0 / dt

        if not np.isfinite(fs) or fs <= 0:
            print(f'Skipping {event_name}: invalid sampling rate')
            continue

        timeseries = TimeSeries(waveform, sample_rate=fs)
        psd = timeseries.psd(
            fftlength=4,
            overlap=2,
            method='median',
            window='hann',
        )

        metrics = xoptimalsnr(
            h=waveform_window[:, np.newaxis],
            t0=0.0,
            fs=fs,
            S=psd.value,
            F0=psd.frequencies.value,
            dF=None,
            Fmin=25,
            Fmax=fs / 2,
        )

        if len(metrics) == 7:
            (
                snr,
                h_rss,
                h_peak,
                fchar,
                bandwidth,
                tchar,
                duration,
            ) = metrics
        else:
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

        result = {
            'event_name': event_name,
            'event_gps': event_gps,
            'template_name': match['template_name'],
            'correlation': match['correlation'],
            'matched_window_start': start,
            'matched_window_end': end,
            'snr': snr,
            'h_rss': h_rss,
            'h_peak': h_peak,
            'fchar': fchar,
            'bandwidth': bandwidth,
            'tchar': tchar,
            'duration': duration,
        }

        output_file = METRIC_DIR / f'{safe_filename(event_name)}.json'
        with output_file.open('w', encoding='utf-8') as file:
            json.dump(
                {
                    key: json_value(value)
                    for key, value in result.items()
                },
                file,
                indent=2,
            )

        print(f'Saved metrics: {output_file}')

        plot_metric_event(
            event_name=event_name,
            event_gps=event_gps,
            start_gps=gps[0],
            waveform_2s=waveform_2s,
            gps_2s=gps_2s,
            waveform_window=waveform_window,
            gps_window=gps_window,
            fs=fs,
            result=result,
        )

        processed += 1
        print(processed)

        if processed >= 30:
            break