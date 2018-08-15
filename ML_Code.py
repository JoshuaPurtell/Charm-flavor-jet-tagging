#import packages
from root_numpy import root2array, tree2array, testdata, array2root, array2tree
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import ROOT
from ROOT import gROOT
np.random.seed(7)


#extract information from ROOT TTrees
file = testdata.get_filepath('out_train_PM.root')
Xtrain = root2array(file, 'tree_train', branches = ['pointing_angle_var','decay_length_var','Drecon_dca_var','trx_dca_var','pk_dca_var1','pk_dca_var2','dca_angle_var','paths_area_var','pk_p_frac_var'])#
Ytrain = root2array(file, 'tree_train', branches = 'validation_var')
Xtrain = Xtrain.view(np.float32).reshape(Xtrain.shape + (-1,))
Ytrain = Ytrain.view(np.int32).reshape(Ytrain.shape + (-1,))
Y_train = [0]
Y_train = Y_train*len(Ytrain)
for i in range(len(Y_train)):
    Y_train[i] = int(Ytrain[i][0])
file = testdata.get_filepath('out_test_PM.root')
Xtest = root2array(file, 'tree_test', branches = ['pointing_angle_var','decay_length_var','Drecon_dca_var','trx_dca_var','pk_dca_var1','pk_dca_var2','dca_angle_var','paths_area_var','pk_p_frac_var'])#['poynting_angle_var','mass_var','decay_length_var','Drecon_dca_var','trx_dca_var','pk_dca_var1','pk_dca_var2']
Ytest = root2array(file, 'tree_test', branches = 'validation_var')
Xtest = Xtest.view(np.float32).reshape(Xtest.shape + (-1,))
Ytest = Ytest.view(np.int32).reshape(Ytest.shape + (-1,))
Y_test = [0]
Y_test = Y_test*len(Ytest)
for i in range(len(Y_test)):
    Y_test[i] = int(Ytest[i][0])
Ztest = Ytest = root2array(file, 'tree_test', branches = ['efficiency_pass_var'])
Ztest = Ytest.view(np.int32).reshape(Ytest.shape + (-1,))

#create net
model = Sequential()
model.add(Dense(14, activation='relu', input_dim=7))
model.add(Dense(8, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#train net
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(Xtrain, Y_train, validation_data=(Xtest,Y_test), epochs=100, batch_size=100, verbose=1)
scores = model.evaluate(Xtest, Y_test)

#print results for insight
print("Error: %.2f%%" % (100-scores[1]*100))
Ypred = model.predict(Xtest)
for i in range(len(Xtest)):
    print("Actual=%i, Predicted=%i" % (Y_test[i], Ypred[i]))
    
