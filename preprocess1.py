import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

def process(path):

	print("preprocess")
	
	df_main = pd.read_csv(path)
	names=list(df_main.columns)

	correlations = df_main.corr()
	# plot correlation matrix
	fig = plt.figure()
##	fig.canvas.set_window_title('Correlation Matrix')
##	ax = fig.add_subplot(111)
##	cax = ax.matshow(correlations, vmin=-1, vmax=1)
##	fig.colorbar(cax)
##	ticks = np.arange(0,9,1)
##	ax.set_xticks(ticks)
##	ax.set_yticks(ticks)
##	ax.set_xticklabels(names)
##	ax.set_yticklabels(names)
##	plt.savefig('results/Correlation Matrix.png')
##	    
##	plt.pause(5)
##	plt.show(block=False)
##	plt.close() 
	
	#scatterplot
	scatter_matrix(df_main)
	    
	plt.savefig('results/Scatter Matrix.png')
	    
	plt.pause(5)
	plt.show(block=False)
	plt.close() 
	
	ncols=3
	plt.clf()
	f = plt.figure(1)
	f.suptitle(" Data Histograms", fontsize=12)
	vlist = list(df_main.columns)
	nrows = len(vlist) // ncols
	if len(vlist) % ncols > 0:
		nrows += 1
	for i, var in enumerate(vlist):
		plt.subplot(nrows, ncols, i+1)
		plt.hist(df_main[var].values, bins=15)
		plt.title(var, fontsize=10)
		plt.tick_params(labelbottom='off', labelleft='off')
	plt.tight_layout()
	plt.subplots_adjust(top=0.88)
	plt.savefig('results/Histogram.png')
	    
	plt.pause(5)
	plt.show(block=False)
	plt.close() 
