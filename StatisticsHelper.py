import time
import matplotlib.pyplot as mp

"""
Helper for decoupling plotting functionality from main sorting and training script.
"""

#Added: Prints at set intervals in script
class PlotHelper:
    def __init__(self,xlabel,ylabel,figInd):
        self.x = []
        self.y = []
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.domain = None
        mp.figure(figInd)

    def addDataPoint(self,n,time):
        self.x.append(n)
        self.y.append(time)

    def plot(self):
        print("Plotting function...")
        if not self.domain:
            self.domain = len(self.x) #arbitrary, could also be y
        else:
            if not self.domain<len(self.x) or not self.domain<len(self.y):
                print("OBS!_____________There must be a Y-value for each X-Value._____________OBS!")
        print("Labeling axes...")
        mp.xlabel(self.xlabel)
        mp.ylabel(self.ylabel)
        print("Labeling axes done.")
        mp.plot(self.x,self.y)
        self.x = []
        self.y = []
        print("Function plot done.")

    def finalize(self):
        print("Showing graph...")
        mp.show()