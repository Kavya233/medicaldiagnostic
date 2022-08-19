import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm

import Preprocess as pre
import CNNTraining as TR
import Predict as PR
import es as DT

import argparse
import cv2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
import glob
import math
import time
import os
import itertools
#import requests
from PIL import Image
from numpy import average, linalg, dot
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from PIL import Image, ImageStat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew

import math
import argparse
import imutils

import pywt
import pywt.data
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import glob
from tensorflow.keras.models import Sequential, load_model

import pre as pr
import Neuralnetwork as nn
import DTALG as dt
import LogisticRegression as lr
import RandomForest as rf
import preprocess1 as pr1
import SVMALG1 as SVM
import NeuralNetwork1 as NN1
import Predict1 as pred1
import Predict1h as pred1h

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"
def heart_diab():
        windh = tk.Tk()
        windh.title("Multi Disease Prediction")

         
        windh.geometry('1280x720')
        windh.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        windh.grid_rowconfigure(0, weight=1)
        windh.grid_columnconfigure(0, weight=1)

        message13 = tk.Label(windh, text="Multi Disease Prediction" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message13.place(x=100, y=10)

        lbl13 = tk.Label(windh, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl13.place(x=10, y=150)

        txt13 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt13.place(x=300, y=165)


        lbl113 = tk.Label(windh, text="Gender",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl113.place(x=10, y=200)

        txt113 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt113.place(x=300, y=215)

        lbl213 = tk.Label(windh, text="Age",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl213.place(x=10, y=250)

        txt213 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt213.place(x=300, y=265)

        lbl313 = tk.Label(windh, text="currentSmoker",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl313.place(x=10, y=300)

        txt313 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt313.place(x=300, y=315)

        lbl413 = tk.Label(windh, text="cigsPerDay",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl413.place(x=10, y=350)

        txt413 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt413.place(x=300, y=365)

        lbl513 = tk.Label(windh, text="BPMeds",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl513.place(x=10, y=400)

        txt513 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt513.place(x=300, y=415)



        lbl613 = tk.Label(windh, text="prevalentStroke",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl613.place(x=10, y=450)

        txt613 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt613.place(x=300, y=465)

        lbl713 = tk.Label(windh, text="prevalentHyp",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl713.place(x=10, y=500)

        txt713 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt713.place(x=300, y=515)

        lbl813 = tk.Label(windh, text="diabetes",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl813.place(x=400, y=200)

        txt813 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt813.place(x=600, y=215)

        lbl913 = tk.Label(windh, text="totChol",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl913.place(x=400, y=250)

        txt913 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt913.place(x=600, y=265)

        lbl1013 = tk.Label(windh, text="sysBP",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1013.place(x=400, y=300)

        txt1013 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1013.place(x=600, y=315)


        lbl1113 = tk.Label(windh, text="DIABP",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1113.place(x=400, y=350)

        txt1113 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1113.place(x=600, y=365)
        lbl1213 = tk.Label(windh, text="BMI",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1213.place(x=400, y=400)

        txt1213 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1213.place(x=600, y=415)
        lbl1313 = tk.Label(windh, text="Heat rate",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1313.place(x=400, y=450)

        txt1313 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1313.place(x=600, y=465)
        lbl1413 = tk.Label(windh, text="glucose",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1413.place(x=400, y=500)

        txt1413 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1413.place(x=600, y=515)
        lbl1513 = tk.Label(windh, text="Result",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1513.place(x=400, y=550)

        txt1513 = tk.Entry(windh,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1513.place(x=600, y=565)

        def hclear():
                txt13.delete(0, 'end') 
                txt113.delete(0, 'end') 
                txt213.delete(0, 'end') 
                txt313.delete(0, 'end') 
                txt413.delete(0, 'end') 
                txt513.delete(0, 'end') 
                txt613.delete(0, 'end') 
                txt713.delete(0, 'end') 
                txt813.delete(0, 'end') 
                txt913.delete(0, 'end') 
                txt1013.delete(0, 'end') 
                txt1113.delete(0, 'end')
                txt1213.delete(0, 'end')
                txt1313.delete(0, 'end')
                txt1413.delete(0, 'end')
                txt1513.delete(0, 'end')

        def hbrowse():
                path=filedialog.askopenfilename()
                print(path)
                txt13.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=wind)    

        
        def hpredictprocess():
                sym=txt13.get()
                txt1513.delete(0, 'end') 
                a1=txt113.get()
                if a1=="F":
                        a1=0
                else:
                        a1=1
                a2=txt213.get()
                a3=txt313.get()
                if a3=="N":
                        a3=0
                else:
                        a3=1
                a4=txt413.get()
                a5=txt513.get()
                a6=txt613.get()
                a7=txt713.get()
                a8=txt813.get()
                a9=txt913.get()
                a10=txt1013.get()
                a11=txt1113.get()
                a12=txt1213.get()
                a13=txt1313.get()
                a14=txt1413.get()
                
                if sym == "":
                        tm.showinfo("Input error", "Select Dataset",parent=windh)
                elif a1 == "":
                        tm.showinfo("Insert error", "Enter Sex",parent=windh)
                elif a2 == "":
                        tm.showinfo("Insert error", "Enter Gender",parent=windh)
                elif a3 == "":
                        tm.showinfo("Insert error", "Enter currentSmoker",parent=windh)
                elif a4 == "":
                        tm.showinfo("Insert error", "Enter cigsPerDay",parent=windh)
                elif a5 == "":
                        tm.showinfo("Insert error", "Enter BPMeds",parent=windh)
                elif a6 == "":
                        tm.showinfo("Insert error", "Enter prevalentStroke",parent=windh)
                elif a7 == "":
                        tm.showinfo("Insert error", "Enter prevalentHyp",parent=windh)
                elif a8 == "":
                        tm.showinfo("Insert error", "Enter Total diabetes",parent=windh)
                elif a9 == "":
                        tm.showinfo("Insert error", "Enter totChol",parent=windh)
                elif a10=="":
                        tm.showinfo("Insert error", "Enter sysBP",parent=windh)
                elif a11=="":
                        tm.showinfo("Insert error", "Enter diaBP",parent=windh)
                elif a12=="":
                        tm.showinfo("Insert error", "Enter BMI",parent=windh)
                elif a13=="":
                        tm.showinfo("Insert error", "Enter heartRate",parent=windh)
                elif a14=="":
                        tm.showinfo("Insert error", "Enter glucose",parent=windh)
                else:
                        new_pred = pred1h.process([float(a1),float(a2),float(a3),float(a4),float(a5),float(a6),float(a7),float(a8),float(a9),float(a10),float(a11),float(a12),float(a13),float(a14)])
                        res=new_pred[0]
                        print("result=",res)
                        txt1513.insert('end', "Predicted as "+str(new_pred))
                        tm.showinfo("Output", "Predicted as "+str(new_pred),parent=windh)
                        

                

                        
        clearButton13 = tk.Button(windh, text="Clear", command=hclear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton13.place(x=960, y=150)

        browse13 = tk.Button(windh, text="Browse", command=hbrowse  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse13.place(x=530, y=150)

        pred13 = tk.Button(windh, text="Predict", command=hpredictprocess  ,fg=fgcolor,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        pred13.place(x=400, y=600)
        pred13 = tk.Button(windh, text="back", command=windh.destroy  ,fg=fgcolor,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        pred13.place(x=600, y=600)

        quitWindow13 = tk.Button(windh, text="QUIT", command=windh.destroy  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow13.place(x=880, y=600)

         
        windh.mainloop()
        
def liver():
        wind = tk.Tk()
        wind.title("Liver Disease Prediction")

         
        wind.geometry('1280x720')
        wind.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        wind.grid_rowconfigure(0, weight=1)
        wind.grid_columnconfigure(0, weight=1)

        message13 = tk.Label(wind, text="Liver Disease Prediction" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message13.place(x=100, y=10)

        lbl13 = tk.Label(wind, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl13.place(x=10, y=200)

        txt13 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt13.place(x=300, y=215)


        lbl113 = tk.Label(wind, text="Age",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl113.place(x=10, y=270)

        txt113 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt113.place(x=300, y=285)

        lbl213 = tk.Label(wind, text="Gender",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl213.place(x=10, y=320)

        txt213 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt213.place(x=300, y=335)

        lbl313 = tk.Label(wind, text="Total Bilirubin",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl313.place(x=10, y=370)

        txt313 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt313.place(x=300, y=385)

        lbl413 = tk.Label(wind, text="Direct Bilirubin",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl413.place(x=10, y=420)

        txt413 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt413.place(x=300, y=435)

        lbl513 = tk.Label(wind, text="Alkaline Phosphotase",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl513.place(x=10, y=470)

        txt513 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt513.place(x=300, y=485)



        lbl613 = tk.Label(wind, text="Alamine Aminotransferase",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl613.place(x=600, y=270)

        txt613 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt613.place(x=860, y=285)

        lbl713 = tk.Label(wind, text="Aspartate Aminotransferase",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl713.place(x=600, y=320)

        txt713 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt713.place(x=860, y=335)

        lbl813 = tk.Label(wind, text="Total Protiens",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl813.place(x=600, y=370)

        txt813 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt813.place(x=860, y=385)

        lbl913 = tk.Label(wind, text="Albumin",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl913.place(x=600, y=420)

        txt913 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt913.place(x=860, y=435)

        lbl1013 = tk.Label(wind, text="Albumin and Globulin_Ratio",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1013.place(x=600, y=470)

        txt1013 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1013.place(x=860, y=485)


        lbl1113 = tk.Label(wind, text="Result",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1113.place(x=400, y=520)

        txt1113 = tk.Entry(wind,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1113.place(x=600, y=535)

        def kedclear():
                txt13.delete(0, 'end') 
                txt113.delete(0, 'end') 
                txt213.delete(0, 'end') 
                txt313.delete(0, 'end') 
                txt413.delete(0, 'end') 
                txt513.delete(0, 'end') 
                txt613.delete(0, 'end') 
                txt713.delete(0, 'end') 
                txt813.delete(0, 'end') 
                txt913.delete(0, 'end') 
                txt1013.delete(0, 'end') 
                txt1113.delete(0, 'end') 

        def kedbrowse():
                path=filedialog.askopenfilename()
                print(path)
                txt13.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=wind)    

        def kedpreprocess2():
                sym=txt13.get()
                if sym != "" :
                        pr.process(sym)
                        tm.showinfo("Input", "Preprocess Finished Successfully",parent=wind)
                else:
                        tm.showinfo("Input error", "Select Dataset")

        def kedSVMprocess():
                sym=txt13.get()
                if sym != "" :
                        SVM.process(sym)
                        tm.showinfo("Input", "SVM Finished Successfully",parent=wind)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=wind)
        def kedNNprocess():
                sym=txt13.get()
                if sym != "" :
                        NN1.process(sym)
                        tm.showinfo("Input", "Neural Network Finished Successfully",parent=wind)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=wind)
        def kedpredictprocess():
                sym=txt13.get()
                txt1113.delete(0, 'end') 
                a1=txt113.get()
                a2=txt213.get()
                if a2=="F":
                        a2=0
                else:
                        a2=1
                a3=txt313.get()
                a4=txt413.get()
                a5=txt513.get()
                a6=txt613.get()
                a7=txt713.get()
                a8=txt813.get()
                a9=txt913.get()
                a10=txt1013.get()
                if sym == "":
                        tm.showinfo("Input error", "Select Dataset",parent=wind)
                elif a1 == "":
                        tm.showinfo("Insert error", "Enter Age",parent=wind)
                elif a2 == "":
                        tm.showinfo("Insert error", "Enter Sex",parent=wind)
                elif a3 == "":
                        tm.showinfo("Insert error", "Enter Total Bilirubin",parent=wind)
                elif a4 == "":
                        tm.showinfo("Insert error", "Enter Direct Bilirubin",parent=wind)
                elif a5 == "":
                        tm.showinfo("Insert error", "Enter Alkaline Phosphotase",parent=wind)
                elif a6 == "":
                        tm.showinfo("Insert error", "Enter Alamine Aminotransferase",parent=wind)
                elif a7 == "":
                        tm.showinfo("Insert error", "Enter Aspartate Aminotransferase",parent=wind)
                elif a8 == "":
                        tm.showinfo("Insert error", "Enter Total Protiens",parent=wind)
                elif a9 == "":
                        tm.showinfo("Insert error", "Enter Albumin",parent=wind)
                elif a10=="":
                        tm.showinfo("Insert error", "Enter Albumin_and_Globulin_Ratio",parent=wind)
                else:
                        new_pred = pred1.process([float(a1),float(a2),float(a3),float(a4),float(a5),float(a6),float(a7),float(a8),float(a9),float(a10)])
                        res=new_pred[0]
                        print("result=",res)
                        txt1113.insert('end', "Predicted as "+str(new_pred))
                        tm.showinfo("Output", "Predicted as "+str(new_pred),parent=wind)
                        

                

                        
        clearButton13 = tk.Button(wind, text="Clear", command=kedclear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton13.place(x=960, y=200)

        browse13 = tk.Button(wind, text="Browse", command=kedbrowse  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse13.place(x=530, y=205)

        #pre13 = tk.Button(wind, text="Preprocess", command=kedpreprocess2  ,fg=fgcolor  ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #pre13.place(x=140, y=600)

        #texta13 = tk.Button(wind, text="SVM", command=kedSVMprocess  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #texta13.place(x=350, y=600)

        #texta113 = tk.Button(wind, text="NeuralNetwork", command=kedNNprocess  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #texta113.place(x=560, y=600)


        pred13 = tk.Button(wind, text="Predict", command=kedpredictprocess  ,fg=fgcolor,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        pred13.place(x=350, y=600)

        quitWindow13 = tk.Button(wind, text="Next", command=heart_diab  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow13.place(x=560, y=600)

         
        wind.mainloop()
def kidney():
        win = tk.Tk()
        win.title("Kidney Disease Prediction")

         
        win.geometry('1280x720')
        win.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        message11 = tk.Label(win, text="Kidney Disease Prediction" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message11.place(x=100, y=10)

        lbl11 = tk.Label(win, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl11.place(x=10, y=200)

        txt11 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt11.place(x=300, y=215)


        lbl112 = tk.Label(win, text="Specific gravity",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl112.place(x=10, y=270)

        txt112 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt112.place(x=300, y=285)

        lbl212 = tk.Label(win, text="Al",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl212.place(x=10, y=320)

        txt212 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt212.place(x=300, y=335)

        lbl312 = tk.Label(win, text="Serum Creatinine",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl312.place(x=10, y=370)

        txt312 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt312.place(x=300, y=385)

        lbl412 = tk.Label(win, text="Hemoglobin Level",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl412.place(x=10, y=420)

        txt412 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt412.place(x=300, y=435)

        lbl512 = tk.Label(win, text="Packed Cell Volume(PCV)",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl512.place(x=10, y=470)

        txt512 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt512.place(x=300, y=485)



        lbl612 = tk.Label(win, text="Renal Hypertension",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl612.place(x=600, y=270)

        txt612 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt612.place(x=860, y=285)


        lbl712 = tk.Label(win, text="Result",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl712.place(x=600, y=320)

        txt712 = tk.Entry(win,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt712.place(x=860, y=335)

        def clear():
                txt11.delete(0, 'end') 
                txt112.delete(0, 'end') 
                txt212.delete(0, 'end') 
                txt312.delete(0, 'end') 
                txt412.delete(0, 'end') 
                txt512.delete(0, 'end') 
                txt612.delete(0, 'end') 
                txt712.delete(0, 'end')
        def preprocess():
                sym=txt11.get()
                if sym != "" :
                        pr.process(sym)
                        tm.showinfo("Input", "Preprocess Finished Successfully",parent=win)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=win)
                

        def browse():
                path=filedialog.askopenfilename()
                print(path)
                txt11.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=win)    

        

        def ModelTraining():
                sym=txt11.get()
                if sym != "" :
                        nn.process(sym)
                        tm.showinfo("Input", "Model Training Finished Successfully",parent=win)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=win)
        def rfprocess():
                sym=txt11.get()
                if sym != "" :
                        rf.process(sym)
                        tm.showinfo("Input", "Random Forest Finished Successfully",parent=win)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=win)
        def lrprocess():
                sym=txt11.get()
                if sym != "" :
                        lr.process(sym)
                        tm.showinfo("Input", "Logistic regression Finished Successfully",parent=win)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=win)
        def dtprocess():
                sym=txt11.get()
                if sym != "" :
                        dt.process(sym)
                        tm.showinfo("Input", "Decision Tree Finished Successfully",parent=win)
                else:
                        tm.showinfo("Input error", "Select Dataset",parent=win)
                

        def predictprocess1():
                sym=txt11.get()
                txt712.delete(0, 'end') 
                a1=txt112.get()
                a2=txt212.get()
                a3=txt312.get()
                a4=txt412.get()
                a5=txt512.get()
                a6=txt612.get()
                
                if sym == "":
                        tm.showinfo("Input error", "Select Dataset",parent=win)
                elif a1 == "":
                        tm.showinfo("Insert error", "Specific gravity",parent=win)
                elif a2 == "":
                        tm.showinfo("Insert error", "AL",parent=win)
                elif a3 == "":
                        tm.showinfo("Insert error", "Serum Creatinine",parent=win)
                elif a4 == "":
                        tm.showinfo("Insert error", "Hemoglobin Level",parent=win)
                elif a5 == "":
                        tm.showinfo("Insert error", "Packed Cell Volume(PCV)",parent=win)
                elif a6 == "":
                        tm.showinfo("Insert error", "Renal Hypertension",parent=win)
                
                else:
                        
                        #print("Model file: ", model_file)
                        df=pd.read_csv(sym)
                        X = df.drop(["classification"], axis=1)
                        y = df["classification"]
                        X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=True)
                        model2=LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
                        model2.fit(X_train, y_train)
                        X_test=[float(a1),float(a2),float(a3),float(a4),float(a5),float(a6)]
                        x_text=np.array(X_test)
                        x_text=x_text.reshape(1, -1)
                        y_pred = model2.predict(x_text)
                        print("Prediction==",y_pred)
                        res=""
                        if y_pred[0]==0:
                                res="No disease"
                        else:
                                res="Kidney Disease"
                        
                        txt712.insert('end', "class " + str(res))
                        
                        
        clearButton12 = tk.Button(win, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton12.place(x=960, y=200)

        browse12 = tk.Button(win, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse12.place(x=530, y=205)

        #pre = tk.Button(win, text="Preprocess", command=preprocess  ,fg=fgcolor  ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #pre.place(x=140, y=600)

        #texta12 = tk.Button(win, text="Model Training", command=ModelTraining  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #texta12.place(x=350, y=600)

        pred12 = tk.Button(win, text="Predict", command=predictprocess1  ,fg=fgcolor,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        pred12.place(x=560, y=600)
        pred112 = tk.Button(win, text="Liver", command=liver  ,fg=fgcolor,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        pred112.place(x=760, y=600)

        #quitWindow12 = tk.Button(win, text="Random Forest", command=rfprocess  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #quitWindow12.place(x=760, y=450)
        quitWindow312 = tk.Button(win, text="Training", command=lrprocess  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow312.place(x=350, y=600)
        #quitWindow212 = tk.Button(win, text="Decision Tree", command=dtprocess  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        #quitWindow212.place(x=960, y=450)
        quitWindow = tk.Button(win, text="Back", command=win.destroy  ,fg=fgcolor ,bg=bgcolor1  ,width=18  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=980, y=600)

         
        win.mainloop()



def Home():
        global window
        def clear():
            print("Clear1")
            txt.delete(0, 'end')    
            txt1.delete(0, 'end')    
            txt2.delete(0, 'end')    
            txt3.delete(0, 'end')    



        window = tk.Tk()
        window.title("Brain Tumor Prediction")

 
        window.geometry('1280x720')
        window.configure(background=bgcolor)
        #window.attributes('-fullscreen', True)

        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        

        message1 = tk.Label(window, text="Medical Disease Detection" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
        message1.place(x=100, y=20)

        lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl.place(x=100, y=200)
        
        txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt.place(x=400, y=215)

        lbl1 = tk.Label(window, text="Select Train Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl1.place(x=100, y=270)
        
        txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt1.place(x=400, y=275)

        lbl2 = tk.Label(window, text="Select Test Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl2.place(x=100, y=340)
        
        txt2 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt2.place(x=400, y=345)

        lbl3 = tk.Label(window, text="Select Image",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl3.place(x=100, y=420)
        
        txt3 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt3.place(x=400, y=425)
        lbl4 = tk.Label(window, text="Enter Patient Name",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
        lbl4.place(x=100, y=500)
        
        txt4 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
        txt4.place(x=400, y=505)

        def browse():
                path=filedialog.askdirectory()
                print(path)
                txt.delete(0, 'end')
                txt.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select DataSet Folder")     

        def browse1():
                path=filedialog.askdirectory()
                print(path)
                txt1.delete(0, 'end')
                txt1.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Training Dataset Folder")    

        def browse2():
                path=filedialog.askdirectory()
                print(path)
                txt2.delete(0, 'end')
                txt2.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Test Dataset Folder")        

        def browse3():
                path=filedialog.askopenfilename()
                print(path)
                txt3.delete(0, 'end')
                txt3.insert('end',path)
                if path !="":
                        print(path)
                else:
                        tm.showinfo("Input error", "Select Input Image")        

        def preproc():
                sym=txt.get()
                if sym != "" :
                        pre.process(sym)
                        print("preprocess")
                        tm.showinfo("Input", "Preprocess Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset")

        def Trainprocess():
                sym=txt1.get()
                sym1=txt2.get()
                if sym != "" and sym1 != "":
                        TR.process(sym,sym1)
                        tm.showinfo("Input", "Training Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Train and Test Dataset Folder")

        def Predictprocess():
                sym=txt3.get()
                if sym != "" :
                        res=PR.process(sym)
                        result=""
                        if res==0:
                                result="Giloblastma"
                        if res==1:
                                result="astrocytoma"
                        if res==2:
                                result="meningioma"
                        if res==3:
                                result="Pitutry"

                        tm.showinfo("Input", "Class :   "+str(result))
                else:
                        tm.showinfo("Input error", "Select Input Image")
        

        def esprocess():
                sym=txt3.get()
                pname=txt4.get()
                ttype=""
                stime=""
                treat=""
                if sym != "" and pname!= "" :
                        level,lifetime=DT.process(sym)
                        if level=='Begining Tumour':
                                tm.showinfo("Result","You have Diagonose with Begining Tumor")
                                tm.showinfo("Life time","You will survive for another"+str(lifetime))
                                tm.showinfo("Recommended Treatement","Curable after surgery")
                                ttype="Begining Tumour"
                                stime=str(lifetime)+"years"
                                treat="Curable after surgery"
                        if level=='Intermediate Tumour':
                                tm.showinfo("Result","You have Diagonose with Intermediate Tumour")
                                tm.showinfo("Life time","You will survive for another"+str(lifetime))
                                tm.showinfo("Recommended Treatement","Limited Curable after radiation therapy ")
                                ttype="Intermediate Tumour"
                                stime=str(lifetime)+"years"
                                treat="Limited Curable after radiation therapy"
                        if level=='Highlevel Tumour':
                                tm.showinfo("Result","You have Diagonose with Highlevel Tumour")
                                tm.showinfo("Life time","You will survive for another"+str(lifetime))
                                tm.showinfo("Recommended Treatement","Critical non-Curable after chemotherapy ")
                                ttype="Highlevel Tumour"
                                stime=str(lifetime)+"years"
                                treat="Critical non-Curable after chemotherapy"
                        if level=='Normal':
                                tm.showinfo("Result","You have Diagonose with Normal")
                                tm.showinfo("Life time","You will survive for another"+str(lifetime))
                                ttype="No Disease"
                                stime=str(lifetime)+"years"
                                treat="Regular Health Checkup after a year"
                        testdata=[pname,ttype,stime,treat]
                        path=pname+"result.csv"
                        with open(path, 'w') as out_file:
                                writer = csv.writer(out_file)
                                writer.writerow(('Patient_Name', 'Tumor_Type', 'Survaival_Time','Traetment'))
                                writer.writerow(testdata)
                        print('File writen')
                                
                        #tm.showinfo("Input", "Prediction Successfully Finished")
                else:
                        tm.showinfo("Input error", "Select Dataset")

        browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse.place(x=650, y=200)

        browse1 = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse1.place(x=650, y=270)

        browse2 = tk.Button(window, text="Browse", command=browse2  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse2.place(x=650, y=340)

        browse3 = tk.Button(window, text="Browse", command=browse3  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
        browse3.place(x=650, y=420)

        clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
        clearButton.place(x=950, y=200)
         
        proc = tk.Button(window, text="Preprocess", command=preproc  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        proc.place(x=160, y=600)
        

        TRbutton = tk.Button(window, text="Training", command=Trainprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        TRbutton.place(x=400, y=600)


        PRbutton = tk.Button(window, text="Prediction", command=Predictprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        PRbutton.place(x=620, y=600)

        DCbutton = tk.Button(window, text="Prediction of life time", command=esprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        DCbutton.place(x=800, y=600)



        quitWindow = tk.Button(window, text="Kidney", command=kidney  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=1030, y=600)

        window.mainloop()
Home()

