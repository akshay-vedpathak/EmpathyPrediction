import random
import numpy
import statistics
'''
Base line predictor, that predicts the median value from the training data
'''
class BaselinePredictor():
    
    def __init__(self):
        self.label = 5
    
    def fit(self,X,Y):
        print('Training Baseline Classifier on '+str(X.shape[0])+' examples')
        self.label = int(statistics.median(Y))
        print('Setting the value of self.label to median of Y')
    
    def predict(self,X):
        preds = [self.label for i in range(len(X))]
        return numpy.array(preds)