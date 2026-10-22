'''
Takes a hveto triggers text file and loads and saves 16s of calib/calib_highpass
data around each trigger file into a hdf5 file 
'''

import os
from pathlib import Path
import argparse 

os.environ["NDS2_CLIENT_GAP_HANDLER"] = "STATIC_HANDLER_NEG_INF"
os.environ["NDS2_CLIENT_ALLOW_DATA_ON_TAPE"] = '1'
from load_data import grab_main_data

#############################################################
'''
- Use the file in correlation.py to narrow down to a list of triggers I want 
- Compare that list against hveto triggers using hveto_comparison.py
- Do the zoomed-in burst PE with that list 
- Do the lockloss_delays plot again with that list 
'''
#######################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='load strain data around Hveto triggers'
    )
    parser.add_argument('--input', required=True, help='Trigger text file')
    parser.add_argument('--output', required=True, help='Output hdf5 file')
    parser.add_argument('--job-index', type=int, required=True)
    parser.add_argument('--num-jobs', type=int, required=True)
    parser.add_argument('--ifo', choices=['H1','L1'], required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, default=31200)

    args = parser.parse_args() 

    if not 0 <= args.job_index < args.num_jobs:
        parser.error('job-index must be between 0 and num-jobs - 1')

    return args

def select_batch(triggers, job_index, num_jobs):
    start = len(triggers) * job_index // num_jobs 
    end = len(triggers) * (job_index + 1) // num_jobs 
    return triggers[start:end]

def load_hveto(filename):
    hveto_file = Path(filename)
    hveto_triggers = set() 

    with hveto_file.open() as f:
        next(f, None)   # Skips the header, which is the name of each column (time, freq, snr, channel)

        for line_number, line in enumerate(f, start=2):
            fields = line.split() 

            if not fields:
                continue
            
            try:
                event_time = float(fields[0])
            except ValueError:
                print(f'Skipping invalid time on line {line_number}: {line.rstrip()}')
                continue 
            
            hveto_triggers.add(event_time) 
    
    return sorted(hveto_triggers)

def main():
    args = parse_args() 

    os.environ["NDS2_CLIENT_GAP_HANDLER"] = "STATIC_HANDLER_NEG_INF"
    os.environ["NDS2_CLIENT_ALLOW_DATA_ON_TAPE"] = "1"

    hveto_triggers = load_hveto(args.input)
    batch = select_batch(hveto_triggers, args.job_index, args.num_jobs)

    output_path = Path(args.output) 
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grab_main_data(batch, str(output_path), args.ifo, args.host, args.port)    # Saves a hdf5 file containing the channel data for 16s around each hveto trigger 
    print(f'Saved {len(batch)} triggers to {args.output}')

if __name__ == '__main__':
    main()