import time
import random
from StatisticsHelper import PlotHelper as pH
from algorithms.sort import merge_sort, insertion_sort

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


#newPlot = pH("input size/training time(min)","running time(s)/accuracy(%)")

newPlot = pH("input size","running time(s)")

test = TestHelper(1000)
test.queue_sort(merge_sort)
test.queue_sort(insertion_sort)

for algo in test.algorithms:
    for array in test.unordered:
        start_time = time.time()
        algo(array)
        duration = time.time() - start_time
        newPlot.addDataPoint(len(array),duration)
    newPlot.plot()

#newPlot.plot()
#print("Domain is currently {}".format(newPlot.domain))
#time.sleep(5)
newPlot.finalize()