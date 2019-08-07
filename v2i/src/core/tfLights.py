import numpy as np
import random
import os

from v2i.src.core.common import loadPKL, raiseValueError

class RandomIntervalGenerator:

	def __init__(self, min_dur, max_dur, num_pts, start_index, horizon, num_tries=10):
		self.min_dur = min_dur
		self.max_dur = max_dur
		self.num_pts = num_pts
		self.start_index = start_index
		self.generated_pts = []
		self.mini_intervals = []
		self.num_tries = num_tries
		self.horizon = horizon

	def checkValidInterval(self, low, high):

		if low < high:
			if (high - low) > self.max_dur:
				return True
			else:
				return False
		else:
			return False

	def initGenerator(self):

		low = self.start_index
		high = self.horizon

		for try_index in range(0, self.num_tries):
			t = np.random.randint(low, high)
			#print("Iteration : ", try_index+1)
			#print("Time step ", t)
			dur = np.random.randint(self.min_dur, self.max_dur)
			#print("Duration : ", dur)

			new_low = (low, t - 1 - self.max_dur)
			new_high = (t + dur + 1, high - 1 - self.max_dur)

			#print("Low : ", new_low)
			#print("High : ", new_high)
			res1 = self.checkValidInterval(new_low[0], new_low[1])
			res2 = self.checkValidInterval(new_high[0], new_high[1])
			#print(res1, res2)

			if res1:
				self.mini_intervals.append(new_low)

			if res2:
				self.mini_intervals.append(new_high)

			if (res2 or res1):
				self.generated_pts.append((t, dur))
				#print(self.mini_intervals)
				#print(self.generated_pts)
				break

			#print(self.mini_intervals)
			#print(self.generated_pts)
			#print("----------------------------------------")

		if len(self.generated_pts) == 0:
			mid = int(self.horizon/2.0)
			dur = np.random.randint(self.min_dur, self.max_dur)
			new_low = (low, mid - 1 - self.max_dur)
			new_high = (mid + dur + 1, high - 1 - self.max_dur)

			res1 = self.checkValidInterval(new_low[0], new_low[1])
			res2 = self.checkValidInterval(new_high[0], new_high[1])

			if (res1 == False or res2 == False):
				raise ValueError("Invalid horizon length. Should be at least greater than max Duration")

			self.generated_pts.append((mid, dur))
			self.mini_intervals.append(new_low)
			self.mini_intervals.append(new_high)

	def genPoint(self):

		for i in range(0, self.num_tries):
			interval = random.choice(self.mini_intervals)
			t = np.random.randint(interval[0], interval[1])
			dur = np.random.randint(self.min_dur, self.max_dur)

			new_low = (interval[0], t - 1 - self.max_dur)
			new_high = (t + dur + 1, interval[1] - 1 - self.max_dur)

			res1 = self.checkValidInterval(new_low[0], new_low[1])
			res2 = self.checkValidInterval(new_high[0], new_high[1])

			if res1:
				self.mini_intervals.append(new_low)

			if res2:
				self.mini_intervals.append(new_high)

			if (res2 or res1):
				self.generated_pts.append((t, dur))
				self.mini_intervals.remove(interval)
				break

	def gen(self):

		self.initGenerator()
		#print("Pts : ", obj.generated_pts)
		#print("Intervals : ", obj.mini_intervals)
		for i in range(0, self.num_pts):
			self.genPoint()
		#print("Pts : ", obj.generated_pts)
		#print("Intervals : ", obj.mini_intervals)
		self.generated_pts = sorted(self.generated_pts, key=lambda x:x[0])

		return self.generated_pts

	def reset(self):

		self.mini_intervals = []
		self.generated_pts = []

		return self.generated_pts

class tfController:

	def __init__(self, tPeriod):
		self.tfTrajecDict = self.loadtfTrajecs()
		self.validateConfig(self.tfTrajecDict, tPeriod)
	
	def loadtfTrajecs(self):
		currPath = os.path.realpath(__file__)[:-16]
		return loadPKL(currPath + "/data/tftrajec.pkl")
	
	def validateConfig(self, tfTrajecDict, tPeriod):
		if tfTrajecDict["metadata"]["time-period"] != tPeriod:
			raiseValueError("time-period doesn't match")
	
	def sample(self):
		sampleTrajecs = np.random.randint(0, self.tfTrajecDict["numTrajecs"], size=2)
		trajecs = []
		for tid in sampleTrajecs:
			trajecs.append(self.tfTrajecDict["data"][tid])
		return trajecs
	
	def expandPts(self):
		trajecs = self.sample()

		expandedTrajecs = []
		for trajec in trajecs:
			tmpTrajec = []
			for a in trajec:
				tmpTrajec.append(a[0])
				tmpTrajec.append(a[0] + a[1])
			expandedTrajecs.append(tmpTrajec)
		return expandedTrajecs
	
	def toggle(self, curValue, lane):
		return not curValue[lane]
	
	def addDummytfVehicle(self, laneMap, lane):
		dummyVehicle = self.tfDummyVehicle(laneMap, lane)
		laneMap[lane] = np.append(dummyVehicle, laneMap[lane])
		return laneMap
	
	def tfDummyVehicle(self, laneMap, lane):
		vehicleProp = laneMap[0][0].copy()
		vehicleProp["pos"] = 0.0
		vehicleProp["speed"] = 0.0
		vehicleProp["lane"] = lane
		vehicleProp["agent"] = 2 # Used to identify tf vehicle
		vehicleProp["id"] = -1
		return vehicleProp

	

if __name__ == "__main__":

	max_dur = int(30 / 0.1)
	min_dur = int(15 / 0.1)
	#print("Max Dur : ", max_dur)
	#print("Min Dur : ", min_dur)
	num_pts = 10
	num_tries = 10
	horizon = 10000
	start_index = 1
	obj = RandomIntervalGenerator(min_dur, max_dur, num_pts, start_index, horizon, num_tries)

	obj.reset()
	ans = obj.gen()

	print(ans)
	ans = obj.reset()
	print(ans)
	ans = obj.gen()
	print(ans)