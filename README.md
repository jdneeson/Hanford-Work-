Work to characterise ETM Glitches at LHO 

Search pipeline flowchart: 
1. omicron_triggers.py to get list of omicron triggers between two gps times, as a csv 
2. load_hveto.submit to get h5 shards, loading raw data for each trigger time (calls load_data) 
3. correlation.submit to run the search 
4. merge_correlation.py to merge the search results into one file 
5. run any analyses required 
