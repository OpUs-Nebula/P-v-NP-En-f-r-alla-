import time
import matplotlib.pyplot as mp

"""
Helper for decoupling plotting functionality from main sorting and training script.
"""
class PlotHelper:
	def __init__(self,xlabel,ylabel):
		self.x = []
		self.y = []
		self.xlabel=xlabel
		self.ylabel=ylabel
		self.domain = None

	def addDataPoint(self,n,time):
		self.x.append(n)
		self.y.append(time)

	def plot(self):
		if not self.domain:
			self.domain = len(self.x) #arbitrary, could also be y
		else:
			if not self.domain<len(self.x) or not self.domain<len(self.y):
				print("OBS!_____________There must be a Y-value for each X-Value._____________OBS!")

		mp.xlabel(self.xlabel)
		mp.ylabel(self.ylabel)
		mp.plot(self.x,self.y)
		self.x = []
		self.y = []

	def finalize(self):
		mp.show()