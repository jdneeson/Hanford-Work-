import json
import math
from pathlib import Path

############################################################

my_filename = 'correlation_results_test_set.json'
hveto_filename = 'hveto_REFL_A_LF_OUT.txt' 
hveto_filename_2 = 'hveto_DHARD_Y_OUT.txt'

#############################################################

my_file = Path(my_filename)
hveto_file = Path(hveto_filename)
hveto_file_2 = Path(hveto_filename_2)

with my_file.open() as f:
    data = json.load(f) 

my_triggers = set(data['glitch_names'])

hveto_triggers = set() 
hveto_triggers_2 = set()

with hveto_file.open() as f:
    next(f, None)     # Skips the header, which is just the name of each column (time, freq, snr, channel) 

    for line_number, line in enumerate(f, start=2):
        fields = line.split() 

        if not fields:
            continue
        
        try:
            event_time = float(fields[0])
        except ValueError:
            print(f'Skipping invalid time on line {line_number}: {line.rstrip()}')
            continue 
        
        rounded_time = math.floor(event_time)  # Always rounding down, check if this is appropriate 
        event_name = f'event_{rounded_time}.000000'
        hveto_triggers.add(event_name) 

with hveto_file_2.open() as f:
    next(f, None)     # Skips the header, which is just the name of each column (time, freq, snr, channel) 

    for line_number, line in enumerate(f, start=2):
        fields = line.split() 

        if not fields:
            continue
        
        try:
            event_time = float(fields[0])
        except ValueError:
            print(f'Skipping invalid time on line {line_number}: {line.rstrip()}')
            continue 
        
        rounded_time = math.floor(event_time)  # Always rounding down, check if this is appropriate 
        event_name = f'event_{rounded_time}.000000'
        hveto_triggers_2.add(event_name) 

overlap = my_triggers & hveto_triggers

overlap2 = my_triggers & hveto_triggers_2

overlap3 = overlap & overlap2 

overlap4 = hveto_triggers & hveto_triggers_2

if my_triggers:
    print(f'Percent of my triggers found in hveto: '
          f'{len(overlap) / len(my_triggers):.1%}') 

if hveto_triggers:
    print(f'Percent of hveto triggers found by my search: '
          f'{len(overlap) / len(hveto_triggers):.1%}')

if hveto_triggers_2:
    print(f'Percent of hveto triggers found by my search: '
          f'{len(overlap) / len(hveto_triggers_2):.1%}')