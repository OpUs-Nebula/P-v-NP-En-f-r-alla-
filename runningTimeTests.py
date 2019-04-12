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
"""
Notes:
- X = training time, Y = accuracy
- Plot training acc vs evaluation accuracy for demonstration of overfitting
- Layer depth variation? mebe
"""
def neural_inference_test():
    appData = pd.read_csv("googleplaystore.csv")
    print(appData.shape)

neural_inference_test()