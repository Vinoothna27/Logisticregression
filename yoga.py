# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:08:09 2020

@author: nishi
"""

        
import serial # import Serial Library
import numpy  as np # Import numpy
import matplotlib.pyplot as plt #import matplotlib library
from drawnow import *
import time 
from matplotlib import interactive
interactive(True)
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats 
import sklearn
from sklearn.preprocessing import normalize
from scipy import signal
from scipy import stats
from numpy import linalg as LA
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from scipy.interpolate import CubicSpline
from scipy.fftpack import fft, ifft
import scipy as sc      
from scipy import stats 
from scipy.fftpack import fft
from scipy.signal import spectrogram as sp
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
fname =r"C:\Users\nishi\Desktop\VIn\Yoga/Yoga_TRAIN.txt"
#all_data = [line.rstrip('[').rstrip(']') for line in open(fname)];
all_data=[[float(num) for num in line.rstrip('\n').replace('[',' ').replace(']',' ').split()] for line in open(fname)];

fname1 =r"C:\Users\nishi\Desktop\VIn\Yoga/Yoga_TEST.txt"
#all_data = [line.rstrip('[').rstrip(']') for line in open(fname)];
all_data1=[[float(num) for num in line.rstrip('\n').replace('[',' ').replace(']',' ').split()] for line in open(fname1)];

all_data1= np.asarray(all_data1) 
train=np.empty((426,))
test=np.empty((426,))

train_y=np.empty((1,))
test_y=np.empty((1,))
u=0
for i in range(300):
    if all_data[i][0] == 1:
       u=u+1
    x = all_data[i][1:]
    #new_im.show()
    x=np.asarray(x)
    
    
    x=(x-np.min(x))/(np.max(x)-np.min(x))
    train_y=np.append(train_y,int(all_data[i][0]))
    train = np.vstack((train, x))    
train=train[1:][:] 
train_y=train_y[1:][:] 
    

z=np.zeros((427,1)).T
testex=np.empty((426,))
g=0
test_yex=np.empty((1,))
for i in range(3000):
  if (all_data1[i][0] == 2)  & ( g <=  400) : 
    xx=all_data1[i][1:]
    xx=np.asarray(xx)
    xx=(xx-np.min(xx))/(np.max(xx)-np.min(xx))
    test_yex=np.append(test_yex,int(all_data1[i][0]))
    testex = np.vstack((testex, xx))
    all_data1=np.delete(all_data1,i,0)
    all_data1=np.vstack((all_data1,z))
    g=g+1
testex=testex[1:][:] 
test_yex=test_yex[1:][:] 
c=0


for i in range(300):
  if (train_y[i] == 1) & ( c <=  102): 
    train[i][:]=testex[i][:]
    train_y[i]=test_yex[i]
    c=c+1
for i in range(2598):
    xx=all_data1[i][1:]
    xx=np.asarray(xx)
    
    xx=(xx-np.min(xx))/(np.max(xx)-np.min(xx))
    test_y=np.append(test_y,int(all_data1[i][0]))
    test = np.vstack((test, xx))
test=test[1:][:] 
test_y=test_y[1:][:]    
















x_train = train
x_test = test

test_y=(test_y-np.min(test_y))/(np.max(test_y)-np.min(test_y))
train_y=(train_y-np.min(train_y))/(np.max(train_y)-np.min(train_y))
y_test = test_y.reshape(1,-1).T
y_train = train_y.reshape(1,-1).T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)





# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(penalty="l2")
logisticRegr.fit(x_train, y_train)
# Returns a NumPy Array
# Predict for One Observation (image)
#logisticRegr.predict(x_test[0].reshape(1,-1))

predictions = logisticRegr.predict(x_test)

# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

lr_probs = logisticRegr.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores

lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores

print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs,pos_label=1)
# plot the roc curve for the model

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]


plt.show()
plot_confusion_matrix(logisticRegr, x_test, y_test) 
plt.show() 
#
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import plot_precision_recall_curve
#import matplotlib.pyplot as plt
#
#disp = plot_precision_recall_curve(logisticRegr, x_test, y_test)
#disp.ax_.set_title('2-class Precision-Recall curve: '
#                   )
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, lr_probs,pos_label=1) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])