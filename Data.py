import csv
import pandas as pd
import sklearn as sk
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from dictfeaturizer import *
from emojifeaturizer import *

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, mean_absolute_error, classification_report, f1_score

if __name__ == "__main__":

	file = open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv", encoding='latin-1')
	dataset = pd.read_csv(file)
	X = dataset.loc[:, ['STATUS']]
	D =	dataset.loc[:, ['STATUS']]
	y = dataset.cEXT	
	le = LabelEncoder()
	y = le.fit_transform(dataset['cEXT'])	


	#convert dataset into dictionary
	file = open ("D:\Thesis\mypersonality_final\LIWC_English.csv",encoding='latin-1')
	dictionary = csv.DictReader(file, delimiter=',')
	dict_list = []
	for line in dictionary:
		dict_list.append(line)
	itty = {}	
	i = 0	
	for item in dict_list:
		key = dict_list[i].values()
		value = dict_list[i].values()
		dictionaryX = {key, value}
		i+=1
		for item in dictionaryX:
			lils = (list(item))
			itty[(lils[0])] = lils[1]

  	#Train test split in the data
	def train_test_features(featureset,y):
		"""This function splits the featureset into trainingdata (10\%) and testdata (90\%) along a specified random state divider of 1. 
		Lateron the models will be trained and tested on the trainingdata through 10-fold cross validation. 
		The testdata will be used as a held out dataset for final testing

		parameters:
		featureset = a dataset containing a list of features
		y = the values of the testdata. A string of 0's and 1's indicating the category in which an entrance belongs. 
		"""

		X1,X2,y1,y2 = train_test_split(featureset, y, random_state = 1, train_size = 0.9, test_size = 0.1)
		trainingset = X1
		return (trainingset, testset, y1, y2)
  
  	#KNN
	def KNN_model(featureset, y):
		""" This function creates the machine learning models for the KNN Classifier method. The function optimizes the model through Gridsearch.
		Then, it computes the following results: accuracy, precision, recall, AUC, confusion matrix, F1, MAE.

		parameters:
		featureset = a dataset containing a list of features
		y = the values of the testdata. A string of 0's and 1's indicating the category in which an entrance belongs. 
		"""

		trainingset, testset, y1, y2 = train_test_features(featureset, y)
		model = KNeighborsClassifier()
		k_range = list(range(1,5))
		param_grid = dict(n_neighbors=k_range)
		model = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
		model.fit(trainingset,y1)
		pan = model.best_params_
		pan2= model.best_estimator_
		ss = cross_val_score(model,trainingset,y1, cv = 10)
		pp = model.predict(testset)
		acc = accuracy_score(y2,pp)
		conf = confusion_matrix(y2,pp)
		prec = precision_score(y2,pp)
		rec = recall_score(y2,pp)
		roc = roc_auc_score(y2,pp)
		MAE = mean_absolute_error(y2,pp)
		F1 = f1_score(y2,pp)
		return('KNN results:', pan, pan2, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
 
	#SVMs
	def SVMs_model(featureset,y):
		""" This function creates the machine learning models for the Support Vector Machines method (a Bag of Words Principle). 
		The function optimizes the model through Gridsearch. Then, it computes the following results: accuracy, precision,
		 recall, AUC, confusion matrix, F1, MAE.

		parameters:
		featureset = a dataset containing a list of features
		y = the values of the testdata. A string of 0's and 1's indicating the category in which an entrance belongs. 
		"""

		trainingset, testset, y1, y2 = train_test_features(featureset, y)
		model = svm.SVC()
		param_grid = {'C':[1,10,100,1000], 'kernel':['linear','rbf']}
		model = GridSearchCV(svm.SVC(gamma='scale'),param_grid, refit = True, verbose=2)
		model.fit(trainingset,y1)
		pan = model.best_params_
		pan2 = model.best_score_
		ss = cross_val_score(model,trainingset,y1, cv = 10)
		pp = model.predict(testset)
		acc = accuracy_score(y2,pp)
		conf = confusion_matrix(y2,pp)
		prec = precision_score(y2,pp)
		rec = recall_score(y2,pp)
		roc = roc_auc_score(y2,pp)
		MAE = mean_absolute_error(y2,pp)
		F1 = f1_score(y2,pp)
		return('SVMs results:', pan, pan2,'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
 
   #Logistic Regression
	def LR_model(featureset, y):
		""" This function creates the machine learning models for the Logistic Regression method. 
		The function optimizes the model through Gridsearch. Then, it computes the following results: accuracy, precision,
		 recall, AUC, confusion matrix, F1, MAE.

		parameters:
		featureset = a dataset containing a list of features
		y = the values of the testdata. A string of 0's and 1's indicating the category in which an entrance belongs. 
		"""

		trainingset, testset, y1, y2 = train_test_features(featureset, y)		
		model = LogisticRegression()
		param_grid = {'solver':['liblinear', 'lbfgs', 'saga']}
		model = GridSearchCV(LogisticRegression(random_state=0, max_iter=300),param_grid, cv=10)
		model.fit(trainingset,y1)
		pan = model.best_params_
		pan2 = model.best_score_
		ss = cross_val_score(model,trainingset,y1, cv = 10)
		pp = model.predict(testset)
		acc = accuracy_score(y2,pp)
		conf = confusion_matrix(y2,pp)
		prec = precision_score(y2,pp)
		rec = recall_score(y2,pp)
		roc = roc_auc_score(y2,pp)
		MAE = mean_absolute_error(y2,pp)
		F1 = f1_score(y2,pp)
		return('LR results:', pan, pan2, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
 
  	
  	#emoji features
	print ('EMOJI FEATURIZER')
	emoji = emoji_featurizer(D)
	print (KNN_model(emoji,y))
	print (SVMs_model(emoji, y))
	print (LR_model(emoji, y))
 	
 	 #Bag of Words
	print ('BW FEATURIZER')
	vec = CountVectorizer()
	Z = vec.fit_transform(dataset['STATUS'])  															
	features = (pd.DataFrame(Z.toarray(), columns=vec.get_feature_names()))
	print (KNN_model(features, y))
	print (SVMs_model(features, y))
	print (LR_model(features, y))
  
	#LIWC features
	print ('LIWC FEATURIZER')
	dicty = dictff(X,itty)
	print (KNN_model(dicty, y))
	print (SVMs_model(dicty, y))
	print (LR_model(dicty, y))
 	
	#emoji featurizer+LIWC
	print ('EMODICT')
	emodict = pd.concat([dicty, emoji], axis=1)
	print (KNN_model(emodict, y))
	print (SVMs_model(emodict, y))
	print (LR_model(emodict, y))
 	
	#emoji vectorizer + Bag of Words
	print ('EMO+BW')
	emoCV = pd.concat([features,emoji], axis=1)
	print (KNN_model(emoCV, y))
	print (SVMs_model(emoCV, y))	
	print (LR_model(emoCV, y))
 	
	#LIWC + BW
	print ('LIWC+BW')
	dictCV = pd.concat([dicty,features], axis=1)
	print (KNN_model(dictCV, y))
	print (SVMs_model(dictCV, y))
	print (LR_model(dictCV, y))
 	
	#emojifeaturizer+dictionary+CV
	print ('TRES')
	tres= pd.concat([dictCV,emoji], axis=1)
	print (KNN_model(tres, y))
	print (SVMs_model(tres, y))
	print (LR_model(tres, y))