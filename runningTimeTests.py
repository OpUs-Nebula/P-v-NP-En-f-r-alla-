import time
import random
import pandas as pd
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


def clean_ensamble(columns):
    fncmap = {"Size":clean_sizepref,"Reviews":int}
    for columns in fncmap.keys():
        

def clean_sizepref(x):
    sep = list(x)
    pref = sep.pop()
    consts = {"M":1, "k":0.001}
    if pref in ["M","k"]:
        fin = consts[pref]*float("".join(sep))
        print(fin)
    else:
        fin = 0
    return fin

"""
Notes:
- X = training time, Y = accuracy
- Plot training acc vs evaluation accuracy for demonstration of overfitting
- Layer depth variation? mebe
"""
def neural_inference_test(input_fields):
    input_fields = ["Size","Current Ver","Reviews"]
    appData = pd.read_csv("googleplaystore.csv",parse_dates=True, thousands="k")
    DataPoints = ["Size","Current Ver","Reviews"] #Might add last updated if algorithm for calculating time from data gathering to last updated is easy enough
#solution to size being string: string.split, remove last, join, .asfloat()
    X = appData[DataPoints] #call clean_ensamble during assignment? saves problem of modifying original dataframe.
    X["Size"] = X["Size"].apply(clean_sizepref)
    Y = pd.to_numeric(appData["Rating"])
    print(X.shape)
    print(X.info)
    print(X.dtypes)
    print(Y.dtypes)


input = "Size","Current Ver","Reviews"
neural_inference_test()