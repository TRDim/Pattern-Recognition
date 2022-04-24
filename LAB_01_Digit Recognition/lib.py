# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:23:18 2020

@author: DTryfonopoulos
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
matrix_train = np.loadtxt('D:/Αναγνώριση Προτύπων/LABS/LAB_1/pr_lab1_2020-21_data/train.txt')
matrix_test = np.loadtxt('D:/Αναγνώριση Προτύπων/LABS/LAB_1/pr_lab1_2020-21_data/test.txt')

X_train = matrix_train[:,1::]           
y_train = matrix_train[:,0] 
X_test = matrix_test[:,1::] 
y_test = matrix_test[:,0] 

print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

def show_sample(X, index):
    '''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        index (int): index of digit to show
        
        Used another implementation than the one in the .pynb
    '''
    import matplotlib.pyplot as plt

    num = X[index,:]
    num = np.reshape(num,(16,16))
    plt.imshow(num, cmap = 'gray_r')
    #raise NotImplementedError


def plot_digits_samples(X, y):
    '''Takes a dataset and selects one example from each label and plots it in subplots

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
    '''
    import random 
    fig = plt.figure()
    for i in range(10):
        A = np.where(y ==i)
        print('Number of training set for number:'+np.str(i), len(A[0]))
        B = random.choice(A[0])
        print('Random example picked to be depicted:',B)
        C = np.reshape(X[B,:], (16,16))
        fig.add_subplot(2,5,i+1)
        plt.imshow(C, cmap = 'gray_r')
    #raise NotImplementedError


def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the mean for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select.

    Returns:
        (float): The mean value of the digits for the specified pixels
    '''
    index = np.where(y ==digit)[0]

    mat= [];
    elem = pixel[0]
    
    for i in index:
        X2 = np.reshape(X[i,:], (16,16))
        mat.append(X2[elem,elem])

    mat = np.array(mat)
    print('Number of Elements measured:',mat.shape)
    print('Mean value of elem',[elem,elem],'for digit:',digit,'=', np.mean(mat))

    #raise NotImplementedError



def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
    '''Calculates the variance for all instances of a specific digit at a pixel location

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select
        pixels (tuple of ints): The pixels we need to select

    Returns:
        (float): The variance value of the digits for the specified pixels
    '''
    index = np.where(y ==digit)[0]

    mat= [];
    elem = pixel[0]
    
    for i in index:
        X2 = np.reshape(X[i,:], (16,16))
        mat.append(X2[elem,elem])

    mat = np.array(mat)
    print('Number of Elements measured:',mat.shape)
    print('Variance of elem',[elem,elem],'for digit:',digit,'=', np.var(mat))

    #raise NotImplementedError



def digit_mean(X, y, digit):
    '''Calculates the mean for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The mean value of the digits for every pixel
    '''
    print('Number of 0 set elements:',X[y==digit,].shape[0])
    mat = X[y_train == digit] #Keep All the samples for digit 0 
    
    mat_mean = np.mean(mat, axis = 0) # Shape = 1194,256 -> Take the mean per column (for each pixel)
    return mat_mean
    #raise NotImplementedError



def digit_variance(X, y, digit):
    '''Calculates the variance for all instances of a specific digit

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)
        digit (int): The digit we need to select

    Returns:
        (np.ndarray): The variance value of the digits for every pixel
    '''
    print('Number of 0 set elements:',X[y==digit,].shape[0])
    mat = X[y_train == digit] #Keep All the samples for digit 0 
    
    mat_var = np.var(mat, axis = 0 )
    return mat_var
    #raise NotImplementedError


def euclidean_distance(s, m):
    '''Calculates the euclidean distance between a sample s and a mean template m

    Args:
        s (np.ndarray): Sample (nfeatures)
        m (np.ndarray): Template (nfeatures)

    Returns:
        (float) The Euclidean distance between s and m
    '''
    from numpy import linalg 
    eucl_dist = np.linalg.norm(s-m)
    return eucl_dist
    #raise NotImplementedError


def euclidean_distance_classifier(X, X_mean):
    '''Classifier based on the euclidean distance between samples in X and template vectors in X_mean

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        X_mean (np.ndarray): Digits data (n_classes x nfeatures)

    Returns:
        (np.ndarray) predictions (nsamples)
    '''
    dist=[]
    dist2=[]
    for i in range(X_mean.shape[0]):
        dist.append(np.linalg.norm(X - X_mean[i,]))
        dist2.append(np.mean(dist))


    print('Classification based on the L2 distance:',np.argmin(dist2))
    print('Real Value of digit 101 equals:', y_test[101])
    return np.argmin(dist2)
    #raise NotImplementedError



class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):
    """Classify samples based on the distance from the mean feature value"""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import accuracy_score
    
    
    def __init__(self):
        self.X_mean_ = []


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        self.labels_num = len(set(y))  # Set:  Build an unordered collection of unique elements
                
        #Calculates self.X_mean_ based on the mean feature values in X for each class.
        for i in range(10):
            self.X_mean.append(np.mean(X[np.where(y==i)[0]], axis=0))
        
        #self.X_mean_ becomes a numpy.ndarray of shape 
        self.X_mean = np.array(self.X_mean)
        
        #fit always returns self.
        return self
    
        #raise NotImplementedError
        

    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        """Make predictions for X based on the euclidean distance from self.X_mean_"""
        self.X_predictions = []
        Dist_ = []
        
        for i in range (len(X)):
            for j in range (10):
                Dist_.append(np.linalg.norm(X[i] - self.X_mean[j]))
            self.X_predictions.append(np.argmin(Dist_))
            Dist_=[]
        
        #return self.X_predictions
        #raise NotImplementedError

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        
        """Return accuracy score on the predictions for X based on ground truth y"""
        
        return accuracy_score(y,X)
        
        #raise NotImplementedError


def evaluate_classifier(clf, X, y, folds=5):
    """Returns the 5-fold accuracy for classifier clf on X and y

    Args:
        clf (sklearn.base.BaseEstimator): classifier
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (float): The 5-fold classification score (accuracy)
    """
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(clf,X, y, 
                         cv=KFold(n_splits=folds, random_state=42), 
                         scoring="accuracy")
    print("CV error = %f +-%f" % (1. - np.mean(scores), np.std(scores)))
    #raise NotImplementedError

    
def calculate_priors(X, y):
    """Return the a-priori probabilities for every class

    Args:
        X (np.ndarray): Digits data (nsamples x nfeatures)
        y (np.ndarray): Labels for dataset (nsamples)

    Returns:
        (np.ndarray): (n_classes) Prior probabilities for every class
    """
    aPriory=[]

    for i in range(10):
        aPriory.append(len(X[np.where(y==i)])/ len(X))
    aPriory = np.array(aPriory)
    
    return aPriory
    raise NotImplementedError



from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import accuracy_score

    def __init__(self, use_unit_variance=False):
        
        use_unit_variance =1 ; 
        self.use_unit_variance = use_unit_variance
        

    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Calculates self.X_mean_ based on the mean
        feature values in X for each class.

        self.X_mean_ becomes a numpy.ndarray of shape
        (n_classes, n_features)

        fit always returns self.
        """
        self.X_mean_=[]
        
        for i in range(10):
             self.X_mean_.append(np.mean(X[np.where(y==i)[0]],axis=0))
        self.X_mean_ = np.array(self.X_mean_)
             
        #raise NotImplementedError
        #return self


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        def NB_custom (self, X, X_mean, unit_variance):
            self.X = X
            self.X_mean = X_mean
            self.unit_variance = self.unit_variance + 10**(-100)
            expo = np.exp(-np.sum((np.power(self.X_mean-self.X_mean,2)/(2*self.unit_variance))))
            dvar = 10** (-100)
            return (1/(np.sqrt(2*np.pi)*dvar))* expo
        probab = np.zeros((len(self.X), 10))
        pred = []
        for i in range(len(X)):
            for j in range(10):
                prob = NB_custom(X[i], X_mean_[j], var)
                probab [i][j] = aPriory[j] *prob
            pred.append(np.argmax(probab[i]))
    
        pred = np.array(pred)
        print (probab.shape)
    #raise NotImplementedError

    def score(self, X, y):
        """
        Return accuracy score on the predictions
        for X based on ground truth y
        """
        from sklearn.metrics import accuracy_score
        acc_score = accuracy_score(y, pred)
        #raise NotImplementedError


class PytorchNNModel(BaseEstimator, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        # WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
        # TODO: initialize model, criterion and optimizer
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        #raise NotImplementedError

    def fit(self, X, y):
        # TODO: split X, y in train and validation set and wrap in pytorch dataloaders
        train_loader = ...
        val_loader = ...
        # TODO: Train model
        #raise NotImplementedError

    def predict(self, X):
        # TODO: wrap X in a test loader and evaluate
        test_loader = ...
        raise NotImplementedError

    def score(self, X, y):
        # Return accuracy score.
        #raise NotImplementedError

        
def evaluate_linear_svm_classifier(X, y, folds=5):
    """ Create an svm with linear kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    
    
    svm = SVC(kernel='linear')
    clf =svm
    #svm.fit(X,y)
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError

def evaluate_rbf_svm_classifier(X, y, folds=5):
    """ Create an svm with rbf kernel and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    
    
    svm = SVC(kernel='rbf')
    clf = svm
    #svm.fit(X,y)
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError

    #raise NotImplementedError


def evaluate_knn_classifier(X, y, folds=5):
    """ Create a knn and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    
    knn = KNeighborsClassifier(n_neighbors=4)
    clf=knn
    #knn.fit(X_train,y_train)
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError
#raise NotImplementedError
    

def evaluate_sklearn_nb_classifier(X, y, folds=5):
    """ Create an sklearn naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
        
    gnb = GaussianNB()
    clf=gnb
    #gnb.fit(X,y)
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError

    raise NotImplementedError
    
    
def evaluate_custom_nb_classifier(X, y, folds=5):
    """ Create a custom naive bayes classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    
    customNB =CustomNBClassifier()
    clf =customNB
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError

    #raise NotImplementedError
    
    
def evaluate_euclidean_classifier(X, y, folds=5):
    """ Create a euclidean classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    EC = EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin)
    clf = EC
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError
    raise NotImplementedError
    
def evaluate_nn_classifier(X, y, folds=5):
    """ Create a pytorch nn classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.neural_network import MLPClassifier
    
    nn = MLPClassifier()
    clf = nn
    
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError
    
    #raise NotImplementedError    

    

def evaluate_voting_classifier(X, y, folds=5):
    """ Create a voting ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import model_selection
    from sklearn.model_selection import KFoldx
    from sklearn.model_selection import cross_val_score

    kfold = model_selection.KFold(n_splits=10)
    estimators=[]

    model1 = LogisticRegression()
    estimators.append(('logistic', model1))

    model2 = DecisionTreeClassifier()
    estimators.append(('decTree', model2))

    model3 = SVC()
    estimators.append(('svm', model3))

    #Create Ensemble Model 
    ensemble = VotingClassifier(estimators)
    clf = ensemble 
    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedError
    #raise NotImplementedError
    
    

def evaluate_bagging_classifier(X, y, folds=5):
    """ Create a bagging ensemble classifier and evaluate it using cross-validation
    Calls evaluate_classifier
    """
    from sklearn.ensemble import BaggingClassifier
    from sklearn.svm import SVC
    
    bagging = BaggingClassifier(SVC(gamma='auto'), max_samples=0.5, max_features=0.5)
    clf=bagging

    evaluate_classifier(clf, X, y, folds=5)#raise NotImplementedErrorraise NotImplementedError