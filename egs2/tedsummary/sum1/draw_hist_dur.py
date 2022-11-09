import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import json
import os.path
# Creating dataset
duration_file=open("/ocean/projects/cis210027p/shared/corpora/TEDSummary/get_duration.sh")
store_duration=open("/ocean/projects/cis210027p/shared/corpora/TEDSummary/store_duration.txt")
duration_arr=[line.strip() for line in store_duration]
line_arr=[line.strip() for line in duration_file]
duration_dict={}
for k in range(len(line_arr)):
    duration_dict[line_arr[k].split()[2].split("/")[-1]]=duration_arr[k]

# Creating distribution
wav_scp_file=open("data/train/wav.scp")
valid_wav_scp_file=open("data/valid/wav.scp")
test_wav_scp_file=open("data/test/wav.scp")
wav_scp_arr=[line.split()[1].split("/")[-1] for line in wav_scp_file]
wav_scp_arr=wav_scp_arr+[line.split()[1].split("/")[-1] for line in valid_wav_scp_file]
wav_scp_arr=wav_scp_arr+[line.split()[1].split("/")[-1] for line in test_wav_scp_file]
x=[]

for k in wav_scp_arr:
    time=duration_dict[k]
    time_arr=time.split(":")
    dur=int(time_arr[0])*3600+int(time_arr[1])*60+float(time_arr[2])
    if dur>2000:
        x.append(2000)
    else:
        x.append(dur)

 
# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
 
axs.hist(x, bins = 10)
 
# Show plot
plt.savefig("len_graph_duration.png")
plt.show()
