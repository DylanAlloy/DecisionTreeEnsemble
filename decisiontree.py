from numpy import *

class dtree:

	def __init__(self):
		# create constructor

	def read_data(self,filename):
		fid = open(filename,"r")
		data = []
		d = []
		for line in fid.readlines():
			d.append(line.strip())
		for d1 in d:
			data.append(d1.split(","))
		fid.close()

		self.featureNames = data[0]
		self.featureNames = self.featureNames[:-1]
		data = data[1:]
		self.classes = []
		for d in range(len(data)):
			self.classes.append(data[d][-1])
			data[d] = data[d][:-1]

		return data,self.classes,self.featureNames

	def classify(self,tree,datapoint):

		if type(tree) == type("string"):
			# have reached a leaf
			return tree
		else:
			a = list(tree.keys())[0]
			for i in range(len(self.featureNames)):
				if self.featureNames[i]==a:
					break
			
			try:
				t = tree[a][datapoint[i]]
				return self.classify(t,datapoint)
			except:
				return None

	def classifyAll(self,tree,data):
		results = []
		for i in range(len(data)):
			results.append(self.classify(tree,data[i]))
		return results
       
	def make_tree(self,data,classes,featureNames,maxlevel=-1,level=0):
		""" The main function, which recursively constructs the tree"""

		nData = len(data)
		nFeatures = len(data[0])
		
		try: 
			self.featureNames
		except:
			self.featureNames = featureNames
			
		# list the possible classes
		newClasses = []
		for aclass in classes:
			if newClasses.count(aclass)==0:
				newClasses.append(aclass)

		# compute the default class (and total entropy)
		frequency = zeros(len(newClasses))

		totalEntropy = 0
		totalGain = 0
		index = 0
		for aclass in newClasses:
			frequency[index] = classes.count(aclass)
			totalEntropy += self.calc_entropy(float(frequency[index])/nData)
			totalGain += (float(frequency[index])/nData)**2

			index += 1

		totalGain = 1 - totalEntropy
		default = classes[argmax(frequency)]

		if nData==0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
			# have reached an empty branch
			return default
		elif classes.count(classes[0]) == nData:
			# only 1 class remains
			return classes[0]
		else:

			# choose which feature is best	
			gain = zeros(nFeatures)
			for feature in range(nFeatures):
				g = self.calc_info_gain(data,classes,feature)
				gain[feature] = totalEntropy - g

			bestFeature = argmax(gain)
			tree = {featureNames[bestFeature]:{}}

			# list the values that bestFeature can take
			values = []
			for datapoint in data:
				if values.count(datapoint[bestFeature])==0:
					values.append(datapoint[bestFeature])

			for value in values:
				# find the datapoints with each feature value
				newData = []
				newClasses = []
				index = 0
				for datapoint in data:
					if datapoint[bestFeature]==value:
						if bestFeature==0:
							newdatapoint = datapoint[1:]
							newNames = featureNames[1:]
						elif bestFeature==nFeatures:
							newdatapoint = datapoint[:-1]
							newNames = featureNames[:-1]
						else:
							newdatapoint = datapoint[:bestFeature]
							newdatapoint.extend(datapoint[bestFeature+1:])
							newNames = featureNames[:bestFeature]
							newNames.extend(featureNames[bestFeature+1:])
						newData.append(newdatapoint)
						newClasses.append(classes[index])
					index += 1

				# Now recurse to the next level	
				subtree = self.make_tree(newData,newClasses,newNames,maxlevel,level+1)

				# And on returning, add the subtree on to the tree
				tree[featureNames[bestFeature]][value] = subtree

			return tree

	def printTree(self,tree,str):
    		if type(tree) == dict:
        		print(str, tree.keys()[0])
        		for item in tree.values()[0].keys():
            			print(str, item)
            			self.printTree(tree.values()[0][item], str + "\t")
    		else:
    			print(str, "\t->\t", tree)

	def calc_entropy(self,p):
		if p!=0:
			return -p * log2(p)
		else:
			return 0

	def calc_info_gain(self,data,classes,feature):

		# Calculates the information gain based on entropy
		gain = 0
		nData = len(data)

		# List the values that feature can take
		values = []
		for datapoint in data:
			if values.count(datapoint[feature])==0:
				values.append(datapoint[feature])

		featureCounts = zeros(len(values))
		entropy = zeros(len(values))
		g = zeros(len(values))
		valueIndex = 0
		# Find where those values appear in data[feature] and the corresponding class
		for value in values:
			dataIndex = 0
			newClasses = []
			for datapoint in data:
				if datapoint[feature]==value:
					featureCounts[valueIndex]+=1
					newClasses.append(classes[dataIndex])
				dataIndex += 1

			# Get the values in newClasses
			classValues = []
			for aclass in newClasses:
				if classValues.count(aclass)==0:
					classValues.append(aclass)

			classCounts = zeros(len(classValues))
			classIndex = 0
			for classValue in classValues:
				for aclass in newClasses:
					if aclass == classValue:
						classCounts[classIndex]+=1 
				classIndex += 1
			
			for classIndex in range(len(classValues)):
				entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex])/sum(classCounts))
				g[valueIndex] += (float(classCounts[classIndex])/sum(classCounts))**2

			# computes the entropy
			gain = gain + float(featureCounts[valueIndex])/nData * entropy[valueIndex]
			valueIndex += 1
		return gain	
			
