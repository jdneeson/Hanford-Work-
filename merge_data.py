import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np


def _find_shards(shards_dir: Path, pattern: str = 'glitches_*.h5') -> List[Path]:
    files = list(shards_dir.glob(pattern))
    if not files:
        return []
    # Extract numeric job id if present and sort numerically, otherwise sort by name
    def _key(p: Path):
        m = re.search(r'glitches_(\d+)\.h5$', p.name)
        return int(m.group(1)) if m else p.name
    return sorted(files, key=_key)


def merge_shards(
    shards_dir: Path,
    output_path: Path,
    remove_shards: bool = True,
    shard_paths: Optional[List[Path]] = None,
) -> Path:
    '''
    Merge shard files under `shards_dir` (matching glitches_*.h5) into `output_path`.
    If `shard_paths` supplied, use that exact list (already ordered).
    '''
    if shard_paths is None:
        shard_paths = _find_shards(shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f'No shard files found in {shards_dir}')

    shards = shard_paths

    '''with h5py.File(shards[0], 'r') as f0:
        wformshape = f0['waveforms'].shape[1:]
        coeffshape = f0['coeffs'].shape[1:]
        paramshape = f0.get('input_params').shape[1:] if 'input_params' in f0 else None
        dtype = f0['waveforms'].dtype
        attrs = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in f0.attrs.items()}

    total = 0
    for p in shards:
        with h5py.File(p, 'r') as f:
            total += f['waveforms'].shape[0]'''

    temp_output_path = output_path.with_suffix('.tmp.h5')
    if temp_output_path.exists():
        temp_output_path.unlink()

    event_names = [] 
    seen_events = set()

    all_lockloss_times = []
    lockloss_attrs = None

    with h5py.File(temp_output_path, 'w') as out:
        output_events = out.create_group('events')

        for shard_path in shards:
            with h5py.File(shard_path, 'r') as shard:
                if 'events' in shard:
                    for event_name in shard['events']:
                        if event_name in seen_events:
                            print(f'Skipping duplicate event: {event_name}')
                            continue 
                        
                        shard['events'].copy(event_name, output_events, name=event_name) 

                        seen_events.add(event_name) 
                        event_names.append(event_name) 

                if 'metadata' in shard and 'lockloss_times' in shard['metadata']:
                    lockloss_dataset = shard['metadata']['lockloss_times']
                    all_lockloss_times.extend(lockloss_dataset[:].tolist())

                    if lockloss_attrs is None:
                        lockloss_attrs = dict(lockloss_dataset.attrs)

        merged_lockloss_times = np.unique(np.asarray(all_lockloss_times, dtype=np.float64))

        metadata = out.create_group('metadata')
        output_lockloss = metadata.create_dataset(
            'lockloss_times',
            data=merged_lockloss_times,
        )

        if lockloss_attrs is not None:
            for key, value in lockloss_attrs.items():
                output_lockloss.attrs[key] = value

    # Validate the copied event structure 
    with h5py.File(temp_output_path, 'r') as merged:
        if 'events' not in merged:
            temp_output_path.unlink(missing_ok=True)
            raise RuntimeError('Merged file does not contain an "events" group') 
        
        merged_event_names = list(merged['events']) 

        if set(merged_event_names) != set(event_names):
            temp_output_path.unlink(missing_ok=True) 
            raise RuntimeError('Merged file does not contain the same events')

        for event_name in merged_event_names:
            event = merged['events'][event_name]

            if 'channels' not in event:
                temp_output_path.unlink(missing_ok=True) 
                raise RuntimeError(f'Event {event_name} is missing its "channels" group')
            
            for channel_name in event['channels']:
                channel = event['channels'][channel_name] 

                if 'data' not in channel:
                    temp_output_path.unlink(missing_ok=True)
                    raise RuntimeError(f'Channel {event_name}/{channel_name} is missing either "data" or "gps"')

             
    '''with h5py.File(temp_output_path, 'w') as out:
        out.create_dataset('waveforms', shape=(total, *wformshape), dtype=dtype, compression=compression)
        out.create_dataset('coeffs', shape=(total, *coeffshape), dtype=dtype, compression=compression)
        if paramshape is not None:
            out.create_dataset('input_params', shape=(total, *paramshape), dtype=dtype, compression=compression)

        k = 0
        for p in shards:
            with h5py.File(p, 'r') as f:
                n = f['waveforms'].shape[0]
                out['waveforms'][k : k + n] = f['waveforms'][:]
                out['coeffs'][k : k + n] = f['coeffs'][:]
                if paramshape is not None and 'input_params' in f:
                    out['input_params'][k : k + n] = f['input_params'][:]
                k += n

    # Validate merged temp file before publishing
    with h5py.File(temp_output_path, 'r') as merged:
        ok = True
        ok = ok and merged['waveforms'].shape[0] == total
        ok = ok and merged['coeffs'].shape[0] == total
        if paramshape is not None:
            ok = ok and merged['input_params'].shape[0] == total

        ok = ok and merged['waveforms'].shape[1:] == wformshape
        ok = ok and merged['coeffs'].shape[1:] == coeffshape
        if paramshape is not None:
            ok = ok and merged['input_params'].shape[1:] == paramshape

        ok = ok and merged['waveforms'].dtype == dtype
        ok = ok and merged['coeffs'].dtype == dtype
        if paramshape is not None:
            ok = ok and merged['input_params'].dtype == dtype

    if not ok:
        temp_output_path.unlink(missing_ok=True)
        raise RuntimeError('Merged file failed validation. Shards were NOT deleted.')'''

    temp_output_path.replace(output_path)

    # Write a merged config JSON (single config for the whole dataset)
    cfg_out = {
        'n_events' : len(event_names), 
        'event_names' : event_names, 
        'merged_from_shards' : [p.name for p in shards],
    }
    cfg_dump_path = output_path.with_suffix('.config.json')
    with open(cfg_dump_path, 'w', encoding='ascii') as f:
        json.dump(cfg_out, f, indent=2, default=str)

    if remove_shards:
        for p in shards:
            try:
                p.unlink()
            except Exception:
                pass

    return output_path

if __name__ == '__main__':
    base = Path(__file__).resolve().parent / 'data'
    shards_dir = base / 'shards'
    suffix = datetime.utcnow().strftime('%Y-%m-%d') 
    out = base / f'training_data_{suffix}.h5'
    print(f'Auto-detecting shards in {shards_dir} and merging into {out}')
    merge_shards(shards_dir, out, remove_shards=False)