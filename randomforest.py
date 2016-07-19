import csv
import math
import random
import operator
import numpy as np
from collections import defaultdict

#creating GLOBAL variables
TRAINING_DATA = []	#this will hold the training data
TEST_DATA = []	#this will hold the test data
TEST_DATA_COUNT = 48 #number of data points in TEST_DATA
HEAD = []	#this will hold the values of headers
PERCENT_SPLIT = 20	#percentage of the data split for training and testing
TOTAL_TREE = 25		#number of trees included in Random Forest algorithm
SAMPLE_SIZE_PERCENT = 100	#percentage of the sample taken from the TRAINING_DATA for each decision tree
ATTRIBUTE_SAMPLE_SIZE_PERCENT = 100 #percent will help in random attribute selection at each tree level

#method to load the data from file and return the file data
def loadData(fileName):
    global HEAD
    
    #reading file and get file data
    lines = csv.reader(open(fileName))
    HEAD = next(lines)
    fileData = []	#to hold the file data temporary
    for line in lines:
        fileData.append(line)
        
    fileData = np.array(fileData)	#convert file data to array
    
    splitData(fileData)		#calling splitData method to split training and testing data

#this method to split the file data into training and test set
def splitData(fileData):
    global TRAINING_DATA
    global TEST_DATA
    global TEST_DATA_COUNT
    
    linecount = len(fileData)	#count the total lines in the data set
    splitPoint = round(linecount * (PERCENT_SPLIT/100))
    TRAINING_DATA = fileData[:splitPoint]
    TEST_DATA = fileData[splitPoint:]
    TEST_DATA_COUNT = len(TEST_DATA)

#this method will return the random data from the given data based on given sampleSize
def getRandomData(data, sampleSize):
    tempData = [x for x in range(len(data))]
    #to shuffle the data randomly
    random.shuffle(tempData)
    #empty list object to store randomData
    randomData = []
    #loop tempData from starting point to given sampleSize to get random data
    for i in tempData[:sampleSize]:
        randomData.append(data[i])
    #convert randomData to np format data
    randomData = np.array(randomData)
    return randomData

#this method will return the accuracy of the correctly classified instances
def getAccuracy(c1, c2):
    accuracy = round((c1*100)/c2, 4)
    return accuracy
    
#this method will return the entropy of given attribute index for given data    
def getEntropy(data, attrIndex):
    classIndex = len(HEAD) - 1
    entropyClassObj = {}
    entropyValue = 0
    
    if classIndex == attrIndex:     #this part will be calculated when entropy is calculated for whole dataset
        for line in data:
            if line[classIndex] in entropyClassObj:
                entropyClassObj[line[classIndex]] += 1
            else:
                entropyClassObj[line[classIndex]] = 1
        for eachKey in entropyClassObj:
            #calculate probability of attribute for a class index
            prob_of_attr = entropyClassObj[eachKey] / len(data)
            #log value of probaility
            log_of_prob_of_attr = math.log(entropyClassObj[eachKey] / len(data), 2)
            #calculate entropy value by multiply prob and log of prob and subtract from the previously calculated entropy 
            entropyValue -= prob_of_attr * log_of_prob_of_attr
        return entropyValue
        
    else:   #this part will be calculated when entropy is calculated for a given attribute
        #this loop will calculate count for each categorical 
        #data for the given attribute
        for line in data:   
            if line[attrIndex] in entropyClassObj:
                if line[classIndex] in entropyClassObj[line[attrIndex]]:
                    entropyClassObj[line[attrIndex]][line[classIndex]] += 1
                else:
                    entropyClassObj[line[attrIndex]][line[classIndex]] = 1
            else:
                entropyClassObj[line[attrIndex]] = {}
                entropyClassObj[line[attrIndex]][line[classIndex]] = 1

        entropies = {}
        totalEntropies = {}
        #this loop will calculate the total antropy for the given attribute
        for eachKey in entropyClassObj:
            keytotal = 0
            entropies[eachKey] = 0
            for value in entropyClassObj[eachKey]:
                keytotal += entropyClassObj[eachKey][value]
            for value in entropyClassObj[eachKey]:
                prob_of_attr = entropyClassObj[eachKey][value] / keytotal
                log_of_prob_of_attr = math.log(entropyClassObj[eachKey][value] / keytotal, 2)
                entropies[eachKey] -= prob_of_attr * log_of_prob_of_attr
            totalEntropies[eachKey] = keytotal
            
        for eachEntropy in entropies:
            entropyValue += ((totalEntropies[eachEntropy] / len(data)) * entropies[eachEntropy])
        return entropyValue

#this method will return the maximum information gain attribute index   
def getMaxInformationGainAttrIndex(data, attributes):
    #empty object to store all attributes information gain
    attributesInfoGain = {}
    
    for i in attributes:
        tempVal = getEntropy(data, len(HEAD) - 1) - getEntropy(data, i)
        attributesInfoGain[i] = tempVal
    
    #sort all attributes information gain to get max value
    attributesInfoGain = sorted(attributesInfoGain.items(), key=operator.itemgetter(1))
    #return the index of max information gain attribute
    index = attributesInfoGain[len(attributesInfoGain)-1][0]
    return index

#this method will return the Class Label who are in majority    
def getMajorityClass(data, attrIndex):
    majorityCount = {}
    #loop for each data point from given data
    for row in data:
        if row[attrIndex] in majorityCount:
            majorityCount[row[attrIndex]] += 1
        else:
            majorityCount[row[attrIndex]] = 1
    
    #sort the majorityCount object
   
    sortedMajorityCount = sorted(majorityCount.items(), key=operator.itemgetter(1))
    majorityClass = sortedMajorityCount[len(sortedMajorityCount)-1][0]
    return majorityClass

#this method classifies the given data row for a given tree
def getClassify(tree, row, defaultClass=None):
    if not isinstance(tree, dict):  # if the tree is the last node then return the tree
        return tree
    if not tree:  # if the tree is empty i.e node is empty, return the default class as None
        return defaultClass
    
    attrValues = list(tree.values())[0]     
    attribute_index = list(tree.keys())[0]
    
    instAttrValue = row[attribute_index]
    if instAttrValue not in attrValues:  # this value was not in training data
        return defaultClass
    #recursive call of getClassify method
    return getClassify(attrValues[instAttrValue], row, defaultClass)

#this method will return the class label who is in majority
def getMajorityVotedClass(trees, data):
    #get the majority class and assign it to defaultClass
    defaultClass = getMajorityClass(TEST_DATA, len(HEAD)-1)
    #create empty predictedClassObj object
    predictedClassObj = {}
    
    #loop for each tree and for a given data clasify the store the result into predictedClassObj object
    for i in range(len(trees)):
        predictedClass = getClassify(trees[i], data, defaultClass)
        #this if else will increase the count for each class when correctly classified
        if predictedClass in predictedClassObj:
            predictedClassObj[predictedClass] += 1
        else:
            predictedClassObj[predictedClass] = 1
    
    #sort the predictedClassObj to get the class name who is in majority
    predictedClassObj = sorted(predictedClassObj.items(), key=operator.itemgetter(1))
    
    lastIndex = len(predictedClassObj)-1
    return predictedClassObj[lastIndex][0]

#this method will return the decision tree for the given data
def getDecisionTree(data, index, attributes=None):
    global decisiontree

    if attributes is None:  #this code will occur when no attributes are given
        #it will store all the attributes into the attribures variable
        attributes = [i for i in range(0, len(data[0])) if i != index]
        
    #create empty object to store targetInstances
    targetInstances = {}
    
    #get majority from the class and assign it to defaultClass
    for row in data:
        if row[index] in targetInstances:
            targetInstances[row[index]] += 1
        else:
            targetInstances[row[index]] = 1

    targetInstances = sorted(targetInstances.items(), key=operator.itemgetter(1))
    lastIndex = len(targetInstances)-1
    defaultClass =  targetInstances[lastIndex][0]
    
    #if all the instances belong to the same class return the defaultClass
    if len(targetInstances) == 1:
        return defaultClass
    if len(attributes) == 0:
        return defaultClass    
    
    #if all the instances don't belong to the same class then calculate the 
    #max information gain attribtes index and assign that attribute as the node
    #of the tree
    maxInfoGainIndex = getMaxInformationGainAttrIndex(data, attributes)
    decisionTree = {maxInfoGainIndex : {}}
    
    #now consider attributes except the node just calculated
    leftAttributes = [i for i in attributes if i != maxInfoGainIndex]
    
    if len(leftAttributes) != 0:
        #at each level get the random sample attribute 
        #selection from the attribute list which are left
        random.shuffle(leftAttributes)
        leftAttributes = leftAttributes[:round(len(leftAttributes) * (ATTRIBUTE_SAMPLE_SIZE_PERCENT/100))]
        
    partDataForMaxInfoGain = defaultdict(list)
    
    #each level divide the data into partial data
    for row in data:
        partDataForMaxInfoGain[row[maxInfoGainIndex]].append(row)
        
    for data in partDataForMaxInfoGain:
        #and for each partial data recursivly calculate the decision tree
        #untill all data will be classified
        generateSubTree = getDecisionTree(partDataForMaxInfoGain[data], index, leftAttributes)
        decisionTree[maxInfoGainIndex][data] = generateSubTree
    return decisionTree
        
def applyRandomForest(file):
    #load file data
    loadData(file)
    #empty tree object to hold the forest trees
    trees = {}
    
    #get sample size from TRAINING_DATA
    sampleSize = round(len(TRAINING_DATA) * (SAMPLE_SIZE_PERCENT/100))
    
    #loop for no of trees defined for Random Forest in global variables
    for i in range(0, TOTAL_TREE):
        #get random sample data from TRAINING_DATA of a calculated sampleSize
        tempData = [x for x in range(len(TRAINING_DATA))]
        #to shuffle the data randomly
        random.shuffle(tempData)
        #emplty list object to store randomData
        randomData = []
        #loop tempData from starting point to given sampleSize to get random data
        for j in tempData[:sampleSize]:
            randomData.append(TRAINING_DATA[j])
        #convert randomData to np format data        
        sampleTrainData = np.array(randomData)
        
        #storing all the decision trees for each sampleTrainData
        trees[i] = getDecisionTree(sampleTrainData, len(HEAD)-1)
    
    correctlyClassifiedCount = 0
    
    for data in TEST_DATA:
        actualClass = data[len(HEAD)-1]
        predictedClass = getMajorityVotedClass(trees, data)
        if(predictedClass == actualClass):
            correctlyClassifiedCount +=1

    print('Test data: ',TEST_DATA)
    print("\nFOR DATASET: ",file)
    print("Correctly Classified Accuracy : ", getAccuracy(correctlyClassifiedCount, TEST_DATA_COUNT), "%")


def main(file):
    print("\nAlgorithm: Random Forest")
    print("--------------------------")
    print("Trees in Forest: ", TOTAL_TREE)
    print("Percent Split: ", PERCENT_SPLIT)
    print("Percent Sample Size: ", SAMPLE_SIZE_PERCENT)
    print("--------------------------")    
    applyRandomForest(file)
