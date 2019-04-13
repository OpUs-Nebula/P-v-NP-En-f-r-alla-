import time
import random
import pandas as pd
import numpy as np
from StatisticsHelper import PlotHelper as pH
from algorithms.sort import merge_sort, insertion_sort
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

#Separating graphs between sorting/training might be helpful, as generating enough datapoints for neural networks will
#Be much more tedious than for sorting algorithms.
class TestHelper:
    def __init__(self,sample_size):
        self.unordered = self.generate_lists(sample_size)
        self.algorithms = []

    def generate_lists(self,n):
        return [[random.randint(0,25*k+1) for k in range(0,i)] for i in range(1,n+1)]
    
    def queue_sort(self,func):
        self.algorithms.append(func)

#Sorting graph generation
def sorting_test():
    sortPlot = pH("input size","running time(s)")
    test = TestHelper(1000)
    test.queue_sort(merge_sort)
    test.queue_sort(insertion_sort)

    for algo in test.algorithms:
        for array in test.unordered:
            start_time = time.time()
            algo(array)
            duration = time.time() - start_time
            sortPlot.addDataPoint(len(array),duration)
        sortPlot.plot()
    sortPlot.finalize()

#Neural network graph generation

"""
Note: 

Might be worth packaging clean methods in seperate module
"""
def clean_ensamble(df, columns):
    fncmap = {"Size":clean_sizepref,"Current Ver":clean_ver ,"Reviews":int}
    for mapped in fncmap.keys():
        if mapped in columns:
            df[mapped] = df[mapped].apply(fncmap[mapped])
    return df.dropna()

def clean_sizepref(x):
    sep = list(x)
    pref = sep.pop()
    consts = {"M":1, "k":0.001}
    if pref in ["M","k"]:
        fin = consts[pref]*float("".join(sep))
    else:
        fin = 0
    return fin

def clean_ver(x):
    ret = np.nan
    if not isinstance(x, float):
        comps = list(x) if x != ret else ["V"]
        if comps[0].isdigit():
            trunc = []
            decSep = True
            iterN = 0
            end = False
            while iterN<len(comps) and not end:
                cand = comps[iterN]
                if cand.isdigit(): 
                    trunc.append(cand)
                else:
                    if decSep:
                        trunc.append(".")
                        decSep = False
                    else:
                        if not cand in [".", ","]:
                            end = True
                iterN+=1
            ret = float("".join(trunc))
    return ret

"""
Notes:
- X = training time, Y = accuracy
- Plot training acc vs evaluation accuracy for demonstration of overfitting
- Layer depth variation? mebe
Currently clean fnc mapped(or don't need):
Size, Current Ver, Reviews

"""
def neural_inference_test(input_fields):
    appData = pd.read_csv("googleplaystore.csv",parse_dates=True, thousands="k")
    appData = clean_ensamble(appData.drop(appData.index[10472]),input_fields) #Faulty row, had megabyte spec in reviews.
    
    X = appData[input_fields]
    Y = pd.to_numeric(appData["Rating"])
    print(X.shape)
    print(X.info)
    print(X.dtypes)
    print(Y.dtypes)

#Might be worthwhile to add parameter for Y to test aswell
#Might add last updated if algorithm for calculating time from data gathering to last updated is easy enough
neuralInput = ["Size","Current Ver","Reviews"]
neural_inference_test(neuralInput)