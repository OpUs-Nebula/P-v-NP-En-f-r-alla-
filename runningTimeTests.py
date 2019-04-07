import time
from StatisticsHelper import PlotHelper as pH

newPlot = pH("input size/training time(min)","running time(s)/accuracy(%)")

for i in range(0,20):
	newPlot.addDataPoint(i, i**2)

newPlot.plot()
print("Domain is currently {}".format(newPlot.domain))
time.sleep(5)
newPlot.finalize()