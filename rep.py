# importing pandas as pd 
import pandas as pd
import csv
def process(pname, ttype,stime,treat):
    path=pname+"result.csv"
    res=[pname,ttype,stime,treat]
    file = open(path, 'w', newline ='')
    with file:
        header = ['Patient_Name', 'Tumor_Type', 'Survaival_Time','Traetment']
        writer = csv.DictWriter(file, fieldnames = header)
        writer.writeheader()
        writer.writerow(res) 
	
