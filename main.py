from numpy import *
import decisiontreeweighted
import bagging
from pandas_confusion import BinaryConfusionMatrix
import randomforest

fn = "home/vanoccupanther/Desktop/Masters/1.4ML/L4_Assignment/politics2.csv"
tree = decisiontreeweighted.dtree()
bagger = bagging.bagger()
test,classes,features = tree.read_data(fn)
w = ones((shape(test)[0]),dtype = float)/shape(test)[0]

t=tree.make_tree(test,w,classes,features,1)

y_actu = classes
y_pred = tree.classifyAll(t,test)
print("\nTree Stump Prediction")
print(tree.classifyAll(t,test))
print("\nTrue Classes")
print(classes)

c=bagger.bag(test,classes,features,20)
print("\nBagged Results ")
print(bagger.bagclass(c,test))
binary_confusion_matrix = BinaryConfusionMatrix(y_actu, y_pred)
print(" \nBinary confusion matrix:\n%s" % binary_confusion_matrix)
binary_confusion_matrix.print_stats()
rf = randomforest.main(fn)
