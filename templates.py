import argparse
from pathlib import Path
import json 

import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

from correlation import selected_names
from load_data import iter_events
from matplotlib.ticker import MultipleLocator

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot raw and whitened strain for selected template events."
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("data/O4b_glitches_test.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("template_plots"),
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Optional event names; defaults to correlation.selected_names.",
    )
    parser.add_argument(
        "--windows-output",
        type=Path,
        default=Path("template_windows.json"),
        help="JSON file receiving manually selected windows.",
    )
    return parser.parse_args()


def whiten(data, dt):
    series = TimeSeries(data, sample_rate=1 / dt)
    whitened = series.whiten(
        4,
        2,
        fduration=4,
        method="median",
        window="hann",
    )
    return np.asarray(whitened, dtype=float)


def plot_template(event, output_dir):
    event_name = event["name"]
    event_gps = event["event_gps"]

    raw = event["channels"]["calib"]
    highpass = event["channels"]["calib_highpass"]

    raw_values = np.asarray(raw["values"], dtype=float)
    raw_gps = np.asarray(raw["gps"], dtype=float)

    highpass_values = np.asarray(highpass["values"], dtype=float)
    highpass_gps = np.asarray(highpass["gps"], dtype=float)

    whitened = whiten(highpass_values, raw["dt"])

    # Map whitened samples back to GPS if GWpy crops/shortens the series.
    whitened_gps = highpass_gps[:len(whitened)]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        raw_gps - event_gps,
        raw_values,
        linewidth=0.8,
        color="tab:blue",
    )
    axes[0].set_ylabel("Unwhitened strain")
    axes[0].set_title(
        f"{event_name}  (trigger at {event_gps:.6f} GPS)"
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        whitened_gps - event_gps,
        whitened,
        linewidth=0.8,
        color="tab:red",
    )
    axes[1].axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )
    axes[1].set_ylabel("Whitened highpass strain")
    axes[1].set_xlabel("Time relative to trigger (s)")
    axes[1].grid(alpha=0.25)

    axes[0].set_xlim(0, 2)
    axes[1].set_xlim(0, 2)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.1))
    axes[1].xaxis.set_major_locator(MultipleLocator(0.1))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{event_name}.png"
    fig.savefig(output_path, dpi=150)

    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    names = set(args.names) if args.names else set(selected_names)

    args.windows_output.parent.mkdir(parents=True, exist_ok=True)

    if args.windows_output.exists():
        selections = json.loads(args.windows_output.read_text())
    else:
        selections = {}

    # Add an editable placeholder for every template.
    for event_name in names:
        selections.setdefault(
            event_name,
            {
                "start_s": None,
                "end_s": None,
            },
        )

    args.windows_output.write_text(
        json.dumps(selections, indent=2, sort_keys=True)
    )

    found = set()

    for event in iter_events(str(args.templates)):
        if event["name"] not in names:
            continue

        output_path = plot_template(
            event,
            args.output_dir,
        )
        found.add(event["name"])
        print(f"Saved {output_path}")

    print(f"Created selections file: {args.windows_output}")

    missing = names - found
    if missing:
        raise ValueError(f"Selected events not found: {sorted(missing)}")


if __name__ == "__main__":
    main()