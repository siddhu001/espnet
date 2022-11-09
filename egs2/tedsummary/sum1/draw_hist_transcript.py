import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import json
import os.path
# Creating dataset
line_arr=[line for line in open("data/train/transcript")]
line_arr2=[" ".join(line.split()[1:]) for line in line_arr]
valid_line_arr=[line for line in open("data/valid/transcript")]
valid_line_arr2=[" ".join(line.split()[1:]) for line in valid_line_arr]
test_line_arr=[line for line in open("data/test/transcript")]
test_line_arr2=[" ".join(line.split()[1:]) for line in test_line_arr]
# Creating distribution
abstract_line=line_arr2+valid_line_arr2+test_line_arr2
x=[]

for k in abstract_line:
    if len(k.split())>6000:
        x.append(6000)
    else:
        x.append(len(k.split()))

 
# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
 
axs.hist(x, bins = 10)
 
# Show plot
plt.savefig("len_graph_transcript.png")
plt.show()
