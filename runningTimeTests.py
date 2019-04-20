import time
import random
import pandas as pd
import numpy as np
from StatisticsHelper import PlotHelper as pH
from datacleaning import *
from algorithms.sort import merge_sort, insertion_sort
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

class SortHelper:
    def __init__(self,sample_size):
        self.unordered = self.generate_lists(sample_size)
        self.algorithms = []

    def generate_lists(self,n):
        return [[random.randint(0,25*k+1) for k in range(0,i)] for i in range(1,n+1)]
    
    def queue_sort(self,func):
        self.algorithms.append(func)

class FigureIterator:
    def __init__(self):
        self.CurrentFig = 0

    def increment_ret(self):
        self.CurrentFig+=1
        return self.CurrentFig

FigIter = FigureIterator()

#Sorting graph generation
def sorting_test():
    print("Running sorting test...")
    sortPlot = pH("input size","running time(s)",FigIter.increment_ret())
    test = SortHelper(1000)
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
    print("Sorting test done.")

#Neural network graph generation
"""
Notes:
- https://stackoverflow.com/questions/41308662/how-to-tune-a-mlpregressor
- https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider
- For loop cutting down training sample for each run?
DataFrame columns Currently clean fnc mapped(or don't need):
Size, Current Ver, Reviews
"""

def train_predict_ratings (X_Trn, Y_Trn, X_Val, layers):
    model = MLPRegressor(hidden_layer_sizes=layers)
    model.fit(X_Trn, Y_Trn)   
    Prds = model.predict(X_Val)

    return Prds

def norm_df(target):
    return (target - target.min()) / (target.max() - target.min())

def inverse_df(origin, target):
    return (target * (origin.max() - origin.min())) + origin.min()

def neural_inference_test(input_fields, verbose=False):
    print("Running neural inference test...")
    appData = pd.read_csv("googleplaystore.csv",parse_dates=True, thousands="k")
    appData = clean_ensamble(appData.drop(appData.index[10472]),input_fields) #Faulty row, had megabyte spec in reviews
    appData["Rating"] = pd.to_numeric(appData["Rating"])
    
    norm_columns = input_fields+["Rating"]
    normalized_appData = norm_df(appData[norm_columns]) #Neural network sensetive to normalization

    X = normalized_appData[input_fields]
    Y = normalized_appData["Rating"]

    train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state = 0)
    
    def pred_denorm(layer_depth=1, sample_size=1600):
        norm_pred = train_predict_ratings(train_X.iloc[0:sample_size], train_y.iloc[0:sample_size], val_X, (100,)*layer_depth)
        ground_truth = inverse_df(appData["Rating"],pd.DataFrame(val_y, columns=["Rating"]))
        predictions = inverse_df(appData["Rating"],pd.DataFrame(norm_pred, columns=["Rating"])) 
        return mean_absolute_error(ground_truth, predictions)

    #Sample size vs MAE graph generation
    accPlot = pH("sample size","mean absolute error",FigIter.increment_ret())
    
    #Observed sample size for maximum accuracy: 1500-2000(larger and oscillations around ~0.38 remain constant)
    #Might be worth seperating depth/sample test to different 
    for sample_size in range(1,train_X.shape[0]+1, 20):
        print("Current sample size being evaluated: {}".format(sample_size))
        accPlot.addDataPoint(sample_size,pred_denorm(sample_size=sample_size))

    accPlot.plot()
    accPlot.finalize()

    #Network depth vs mae graph generation:
    depthPlot = pH("layer depth","mean absolute error",FigIter.increment_ret())

    for depth in range(1,40):
        print("layer depth being evaluated: {}".format(depth))
        depthPlot.addDataPoint(depth, pred_denorm(layer_depth=depth))

    depthPlot.plot()
    depthPlot.finalize()

    print("Neural inference test done.")

sorting_test()
neuralInput = ["Size","Current Ver","Reviews"]
neural_inference_test(neuralInput,verbose=True)