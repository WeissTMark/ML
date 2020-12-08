# ML.py
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.optimize
import math
import operator
from numpy import linalg
from numpy import power
from numpy import sqrt
from numpy import mean
from numpy import std
from numpy import sum as arraysum
from matplotlib import pyplot


class Perceptron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter

    """
        In:
            X - data set
            y - training set
        Returns: \

        This function fits the data with a perceptron model
        to be used with other various functions
    """
    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]"""
        self.weight = np.zeros(1+ X.shape[1])
        self.errors = []

        # weights: create a weights array of right size
        # and initialize elements to zero
        # Number of misclassifications (self.errors), creates an array
        # to hold the number of misclassifications

        # main loop to fit the data to the labels
        for _ in range(self.niter):
            # set iteration error to zero
            errors = 0
            # loop over all the objects in X and corresponding y element
                # calculate the needed (delta_w) update from previous step
                # delta_w = rate * (target – prediction current object)
                # calculate what the current object will add to the weight
                # set the bias to be the current delta_w
                # increase the iteration error if delta_w != 0
                # Update the misclassification array with # of errors in iteration
                # return self
            for xi, target in zip(X, y):
                update = self.rate * (target - self.predict(xi))
                self.weight[1:] += update * xi
                self.weight[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
            if errors == 0:
                return self
        return self

    """
        In:
            X - data set
        Returns:
            int - dot product of the weight and bias
    """
    def net_input(self, X):
        """Calculate net input"""
        # return the dot product: X.w + bias

        return np.dot(X, self.weight[1:]) + self.weight[0]

    """
        In:
            X - data set
        Returns:
            array - Part of the input data where the y value at the same index 
            equals a certain value
    """
    def predict(self, X):
        """Return class label after unit step"""
        temp = np.where(self.net_input(X) >= 0, 1, -1)
        return temp

class linReg(object):
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
        
    """
        In:
            x - data set
            y - training set
        Returns: \

        This function fits a linear regression line to the given 
        input data.
    """
    def fit(self, x, y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        xy = np.sum(np.multiply(x, y)) - len(x) * mean_x * mean_y
        xx = np.sum(np.multiply(x, x)) - len(x) * mean_x * mean_x
        
        self.b1 = xy / xx
        self.b0 = mean_y - self.b1 * mean_x
        return self
    
    """
        In:
            inputData - data set (Commonly X)
        Returns:
            int - slope of linear prediction line
    """
    def predict(self, inputData):
        return self.b0 + self.b1 * inputData

    """
        In:
            x - data set
            y - training set
            predictions - predicted linear reg model
            xLoc - The location of the desired interval
        Returns: \

        This creates a scatter plot showing the linear regression line, as well as
        the interval the points could appear in at the given point.
    """
    def interval(self, x, y, predictions, xLoc):
        x_in = x[xLoc]
        #y_out = y[0]
        yhat_out = predictions[xLoc]

        sum_errs = arraysum((y - predictions)**2)
        stdev = sqrt(1/(len(y)-2) * sum_errs)

        interval = 1.96 * stdev

        pyplot.scatter(x, y)
        pyplot.plot(x, predictions, color='red')
        pyplot.errorbar(x_in, yhat_out, yerr=interval, color='black', fmt='o')
        pyplot.show()



class logReg:
    def __init__(self, lr=0.01, niter=100000, fitIntercept=True):
        self.lr = lr
        self.niter = niter
        self.fitIntercept = fitIntercept
    
    """
        In:
            X - data set
        Returns:
            array - Data set with an added X value
    """
    def addIntercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    

    """
        In:
            z - angle
        Returns:
            sigmoid - based off the input angle
    """
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    """
        In:
            h - Sigmoid
            y - Training set
        Returns:
            int - loss value between the two inpuuts
    """
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    """
        In:
            X - Data set
            y - Training set
        Returns: \
            
        This takes the data set, and applies a logistic regression line
        to the set based on the training set
    """
    def fit(self, X, y):
        if self.fitIntercept:
            X = self.addIntercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.niter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
    """
        In:
            X - data set
        Returns:
            int - dot product between X and Theta
    """
    def predict_prob(self, X):
        if self.fitIntercept:
            X = self.addIntercept(X)
    
        return self.sigmoid(np.dot(X, self.theta))
    
    """
        In:
            X - data set
        Returns:
            int - rounded prediction
    """
    def predict(self, X):
        return self.predict_prob(X).round()


       

"""
    In: 
        X - Data set
        y - training set
        classifier - perceptron model
        resolution - How clear the line is (smaller number = more clear)
    Returns: \

    This plots the boundry and color codes the points given from the 
    perceptron to create a clear visualization of the model
"""
def plot_decision_regions(X, y, classifier, resolution):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    

    # plot the decision surface
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1
    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
    np.arange(x2_min, x2_max, resolution))

    array = np.c_[xx1.ravel(), xx2.ravel()]
   
    
    Z = classifier.predict(array)
    Z = Z.reshape(xx1.shape) 

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],   
                    alpha=0.8, 
                    c=np.array([cmap(idx)]), 
                    marker=markers[idx], 
                    label=cl)

"""
    In: 
        x - Data set
        y - training set
        pred - predictions from the linear regression model
    Returns: 
        array - classified points from x

    Using the linear regreshion as a threshold, creates a binary classified array
"""
def thresholdFunc(x, y, pred):
    thresh = np.zeros(x.size)
    for i in x:
        if (y[i] > pred[i]):
            thresh[i] = 1
    return thresh



class SVM(object):
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    

    """
        In: 
            data - array of points
        Returns: \

        This function fits a linear vector to set of points
        and finds the margins between that vector and the points
        themselves
    """
    def fit(self, data):
        #train with data
        self.data = data
        opt_dict = {}
        
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        
        all_data = np.array([])
        for yi in self.data:
            all_data = np.append(all_data,self.data[yi])
                    
        self.max_feature_value = max(all_data)         
        self.min_feature_value = min(all_data)
        all_data = None
        
        #with smaller steps our margins and db will be more precise
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #point of expense
                      self.max_feature_value * 0.001,]
        
        #extremly expensise
        b_range_multiple = 5
        #we dont need to take as small step as w
        b_multiple = 5
        
        latest_optimum = self.max_feature_value*10
        
        """
        objective is to satisfy yi(x.w)+b>=1 for all training dataset such that ||w|| is minimum
        for this we will start with random w, and try to satisfy it with making b bigger and bigger
        """
        
        #making step smaller and smaller to get precise value
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            
            #we can do this because convex
            
            optimized = False
            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        
                        #weakest link in SVM fundamentally
                        #SMO attempts to fix this a bit
                        # ti(xi.w+b) >=1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option=False
                        if found_option:
                            
                            #all points in dataset satisfy y(w.x)+b>=1 for this cuurent w_t, b
                            #then put w,b in dict with ||w|| as key
                            
                            opt_dict[np.linalg.norm(w_t)]=[w_t,b]
                
                #after w[0] or w[1]<0 then values of w starts repeating itself because of transformation
                #Think about it, it is easy
                #print(w,len(opt_dict)) Try printing to understand
                if w[0]<0:
                    optimized=True
                    #print("optimized a step")
                else:
                    w = w-step
                   
            # sorting ||w|| to put the smallest ||w|| at poition 0 
            norms = sorted([n for n in opt_dict])
            #optimal values of w,b
            opt_choice = opt_dict[norms[0]]

            self.w=opt_choice[0]
            self.b=opt_choice[1]
            
            #start with new latest_optimum (initial values for w)
            latest_optimum = opt_choice[0][0]+step*2
            
    """
        In: 
            features - array of points to be predicted
        Returns: 
            int - Predicted classification of points
    """
    def predict(self, features):
        #sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification!=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
        return (classification,np.dot(np.array(features),self.w)+self.b)
    

    """
        In: \
        Returns: \

        This function visualizes the SVM on a graph in
        the lab book
    """
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,c=self.colors[i]) for x in self.data[i]] for i in self.data]
        
        # hyperplane = x.w+b (actually its a line)
        # v = x0.w0+x1.w1+b -> x1 = (v-w[0].x[0]-b)/w1
        #psv = 1     psv line ->  x.w+b = 1a small value of b we will increase it later
        #nsv = -1    nsv line ->  x.w+b = -1
        # dec = 0    db line  ->  x.w+b = 0
        def hyperplane(x,w,b,v):
            #returns a x2 value on line when given x1
            return (-w[0]*x-b+v)/w[1]
       
        hyp_x_min= self.min_feature_value*0.9
        hyp_x_max = self.max_feature_value*1.1
        
        # (w.x+b)=1
        # positive support vector hyperplane
        pav1 = hyperplane(hyp_x_min,self.w,self.b,1)
        pav2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[pav1,pav2],'k')
        
        # (w.x+b)=-1
        # negative support vector hyperplane
        nav1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nav2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nav1,nav2],'k')
        
        # (w.x+b)=0
        # db support vector hyperplane
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')


"""
    In:
        A - a point
        B - a point
    Returns:
        int - The distance between the points
"""
def distance(A, B):
    return np.sum((A - B) ** 2) ** 0.5


"""
    In: 
        X - Data set
        y - Training set
        xQuery - Point in X that we want to classify
        k - The number of neighbors we want to look at

    Returns: 
        int - The suggested classification based on the nearby neighbors
            (This will be a value that is present in y)
"""
def kNN (X, y, xQuery, k=3):
    m = X.shape[0]
    distances = []

    for i in range(m):
        dis = distance(xQuery, X[i])
        distances.append((dis, y[i]))

    distances = sorted(distances)
    distances = distances[:k]

    distances = np.array(distances)
    labels = distances[:,1]

    uniqueLabel, counts = np.unique(labels, return_counts=True)
    pred = uniqueLabel[counts.argmax()]

    return int(pred)