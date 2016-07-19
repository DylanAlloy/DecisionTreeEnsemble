REQ:
 You need pandas_confusion, which includes pandas which you can get at https://pypi.python.org/pypi/pandas_confusion/

QUICK RUNDOWN:
	Gets the information gain using Shannon-entropy down the decision tree leaves.
	Keeps track of nodes to split on and does just that until the IG (info gain) is 1.0 (all one class).
	Stump predictions. It will use this node to infer data about the rest of the classes.
	It will organize the true/false ratio (though it only needs to find the amount that are true) in the final classes.
	The classes will be tuned a little with bootstrap aggregation and a mixed bag approach i.e. a model of random entries in the data that are meant to add some variability to the predictive model in order to train on more than just the data we have. This is especially useful for smaller datasets. We assume a scenario and model based on those assumptions after some intitial training. A confusion matrix then decides which are accurate and some pruning occurs in order to not add noise.
	 
