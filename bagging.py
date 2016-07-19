from numpy import *
import decisiontree

class bagger:

	def __init__(self):
		""" Constructor """
		self.tree = decisiontree.dtree()
		
	def bag(self,data,targets,features,nSamples):
	
		nPoints = shape(data)[0]
		nDim = shape(data)[1]
		self.nSamples = nSamples
		
		# compute boostrap samples
		samplePoints = random.randint(0,nPoints,(nPoints,nSamples))
		classifiers = []
		
		for i in range(nSamples):
			sample = []
			sampleTarget = []
			for j in range(nPoints):
				sample.append(data[samplePoints[j,i]])
				sampleTarget.append(targets[samplePoints[j,i]])
			# train classifiers
			classifiers.append(self.tree.make_tree(sample,sampleTarget,features,1))

		return classifiers
	
	def bagclass(self,classifiers,data):
		
		decision = []
		# majority voting
		for j in range(len(data)):
			outputs = []
			#print data[j]
			for i in range(self.nSamples):
				out = self.tree.classify(classifiers[i],data[j])
				if out is not None:
					outputs.append(out)
			# list the possible outputs
			out = []
			for each in outputs:
				if out.count(each)==0:
					out.append(each)
			frequency = zeros(len(out))
		
			index = 0
			if len(out)>0:
				for each in out:
					frequency[index] = outputs.count(each)
					index += 1
				decision.append(out[frequency.argmax()])
			else:
				decision.append(None)
		return decision
