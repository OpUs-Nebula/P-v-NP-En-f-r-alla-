import time
import random
import pandas as pd
import numpy as np
from statisticshelper import PlotHelper as pH
from datacleaning import *
from algorithms.sort import merge_sort, insertion_sort
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

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
Notes:
- X = depth/sample size(separate graphs), Y = accuracy
- Plot training acc vs evaluation accuracy for demonstration of overfitting
- Layer depth variation? mebe
- https://stackoverflow.com/questions/41308662/how-to-tune-a-mlpregressor
- https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider
- For loop cutting down training sample for each run?
Currently clean fnc mapped(or don't need):
Size, Current Ver, Reviews
"""

def train_predict_ratings (X_Trn, Y_Trn, X_Val):
    model = MLPRegressor()
    model.fit(X_Trn, Y_Trn)   
    Prds = model.predict(X_Val)

    return Prds

def norm_df(target):
    return (target - target.min()) / (target.max() - target.min())

def inverse_df(origin, target):
    return (target * (origin.max() - origin.min())) + origin.min()

def neural_inference_test(input_fields, verbose=False):
    appData = pd.read_csv("googleplaystore.csv",parse_dates=True, thousands="k")
    appData = clean_ensamble(appData.drop(appData.index[10472]),input_fields) #Faulty row, had megabyte spec in reviews
    appData["Rating"] = pd.to_numeric(appData["Rating"])
    
    norm_columns = input_fields+["Rating"]
    normalized_appData = norm_df(appData[norm_columns]) #Neural network sensetive to normalization

    #print("Dataset:")
    #print(appData[norm_columns])

    X = normalized_appData[input_fields]
    Y = normalized_appData["Rating"]

    train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 0)

    #if verbose:
    #    train_size = train_X.shape
    #    print("Shape training sample: x={}, y={}".format(train_size[0],train_y.shape[0]))
    #    print(train_X.iloc[train_size[0]-1])
    
    accPlot = pH("sample size","mean absolute error")

    for sample_size in range(1,train_X.shape[0]+1, 20):
        norm_pred = train_predict_ratings(train_X.iloc[0:sample_size], train_y.iloc[0:sample_size], val_X)
        ground_truth = inverse_df(appData["Rating"],pd.DataFrame(val_y, columns=["Rating"]))
        predictions = inverse_df(appData["Rating"],pd.DataFrame(norm_pred, columns=["Rating"])) 
        mae = mean_absolute_error(ground_truth, predictions)
        accPlot.addDataPoint(sample_size, mae)
        if sample_size % 1 == 0:
            print("Current sample size being evaluated: {}".format(sample_size))

    accPlot.plot()
    accPlot.finalize()
    
#Moved into for loop:
    #norm_pred = train_predict_ratings(train_X, train_y, val_X)
    #ground_truth = inverse_df(appData["Rating"],pd.DataFrame(val_y, columns=["Rating"]))
    #predictions = inverse_df(appData["Rating"],pd.DataFrame(norm_pred, columns=["Rating"])) 
    #mae = mean_absolute_error(ground_truth, predictions)

    #if verbose:
        #print("To be predicted:")
        #print(ground_truth)

        #print("Resulting predictions:")
        #print(predictions)

    #To be replaced by graph
    #print("Mean Absolute Error is: {}".format(mae))

#Might be worthwhile to add parameter for Y to test aswell
#Might add last updated if algorithm for calculating time from data gathering to last updated is easy enough
neuralInput = ["Size","Current Ver","Reviews"]
neural_inference_test(neuralInput,verbose=True)