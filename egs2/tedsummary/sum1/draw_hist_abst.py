import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import json
import os.path
# Creating dataset
line_arr=[line for line in open("data/train/text")]
line_arr2=[" ".join(line.split()[1:]) for line in line_arr]
line_arr3=[line.split(" [SEP] ")[1] for line in line_arr2]
valid_line_arr=[line for line in open("data/valid/text")]
valid_line_arr2=[" ".join(line.split()[1:]) for line in valid_line_arr]
valid_line_arr3=[line.split(" [SEP] ")[1] for line in valid_line_arr2]
test_line_arr=[line for line in open("data/test/text")]
test_line_arr2=[" ".join(line.split()[1:]) for line in test_line_arr]
test_line_arr3=[line.split(" [SEP] ")[1] for line in test_line_arr2]
# Creating distribution
abstract_line=line_arr3+valid_line_arr3+test_line_arr3
x=[]

for k in abstract_line:
     x.append(len(k.split()))

 
# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
 
axs.hist(x, bins = 10)
 
# Show plot
plt.savefig("len_graph_abstract.png")
plt.show()
