import csv
import pandas as pd
import sklearn as sk
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from dictfeaturizer import *
from emojifeaturizer import *

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, mean_absolute_error, classification_report, f1_score

if __name__ == "__main__":

	file = open ("mypersonality_final2.csv", encoding = 'latin-1')
	dataset = pd.read_csv(file)
	X = dataset.loc[:, ['STATUS']]
	y = dataset.cEXT
	le = LabelEncoder()
	y = le.fit_transform(dataset['cEXT'])
	train, test, y1, y2 = train_test_split(X,y, random_state = 1, train_size = 0.9, test_size = 0.1)
	train_l, test_l, y1, y2 = train_test_split(X,y, random_state = 1, train_size = 0.9, test_size = 0.1)
	train_b, test_b, y1, y2 = train_test_split(X,y, random_state = 1, train_size = 0.9, test_size = 0.1)
	
  	#KNN
	def KNN_model(train_f, test_f, y1, y2):
		""" This function creates the machine learning models for the KNN Classifier method. The function optimizes the model through Gridsearch.
		Then, it computes the following results: accuracy, precision, recall, AUC, confusion matrix, F1, MAE, classification report.

		parameters:
		featureset = a data set containing a list of features
		y = the values of the test data. A string of 0's and 1's indicating the category to which an instance belongs.
		"""

		model = KNeighborsClassifier()
		k_range = list(range(1,5))
		param_grid = dict(n_neighbors = k_range)
		cv = StratifiedKFold(n_splits = 10, random_state = 0)
		model = GridSearchCV(model, param_grid, iid = True, cv = cv, scoring = 'f1')
		model.fit(train_f, y1)
		pan= model.best_params_
		ss = cross_val_score(model, train_f, y1, cv = cv, scoring = 'f1_weighted')
		pp = model.predict(test_f)
		acc = accuracy_score(y2, pp)
		conf = confusion_matrix(y2, pp)
		prec = precision_score(y2, pp)
		rec = recall_score(y2, pp)
		roc = roc_auc_score(y2, pp)
		MAE = mean_absolute_error(y2, pp)
		F1 = f1_score(y2, pp, average = 'weighted')
		classy = classification_report(y2, pp)
		return('KNN results:', pan, classy, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )

	#SVMs
	def SVMs_model(train_f, test_f, y1, y2):
		""" This function creates the machine learning models for the Support Vector Machines method (a Bag of Words Principle).
		The function optimizes the model through Gridsearch. Then, it computes the following results: accuracy, precision,
		 recall, AUC, confusion matrix, F1, MAE, classification report.

		parameters:
		featureset = a data set containing a list of features
		y = the values of the test data. A string of 0's and 1's indicating the category to which an instance belongs.
		"""

		model = svm.SVC()
		param_grid = {'C':[1,10]}
		cv = StratifiedKFold(n_splits = 10, random_state = 0)
		model = GridSearchCV(svm.SVC(gamma = 'scale', kernel = 'linear'), param_grid, iid = True, cv = cv, refit = True, verbose = 2, scoring = 'f1')
		model.fit(train_f, y1)
		pan = model.best_params_
		ss = cross_val_score(model, train_f, y1, cv = cv, scoring = 'f1_weighted')
		pp = model.predict(test_f)
		acc = accuracy_score(y2, pp)
		conf = confusion_matrix(y2, pp)
		prec = precision_score(y2, pp)
		rec = recall_score(y2, pp)
		roc = roc_auc_score(y2, pp)
		MAE = mean_absolute_error(y2, pp)
		F1 = f1_score(y2, pp, average = "weighted")
		classy = classification_report(y2, pp)
		return('SVMs results:', pan, classy, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )

   #Logistic Regression
	def LR_model(train_f, test_f, y1, y2):
		""" This function creates the machine learning models for the Logistic Regression method.
		The function optimizes the model through Gridsearch. Then, it computes the following results: accuracy, precision,
		 recall, AUC, confusion matrix, F1, MAE, classification report.

		parameters:
		featureset = a data set containing a list of features
		y = the values of the test data. A string of 0's and 1's indicating the category in which an instance belongs.
		"""

		model = LogisticRegression()
		param_grid = {'C': [1,10], 'solver' : ['liblinear', 'lbfgs', 'saga']}
		cv = StratifiedKFold(n_splits = 10, random_state = 0)
		model = GridSearchCV(LogisticRegression(random_state = 0, max_iter = 900), param_grid, iid = True, cv = cv, scoring = 'f1')
		model.fit(train_f, y1)
		pan = model.best_params_
		ss = cross_val_score(model, train_f, y1, cv = cv, scoring = 'f1_weighted')
		pp = model.predict(test_f)
		acc = accuracy_score(y2, pp)
		conf = confusion_matrix(y2, pp)
		prec = precision_score(y2, pp)
		rec = recall_score(y2, pp)
		roc = roc_auc_score(y2, pp)
		MAE = mean_absolute_error(y2, pp)
		F1 = f1_score(y2, pp, average = "weighted")
		classy = classification_report(y2, pp)
		return('LR results:', pan, classy, 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )

	#emoji features
	print ('EMOJI FEATURIZER')
	emoji_train = emoji_featurizer(train)
	emoji_test = emoji_featurizer(test)
	train_e = emoji_train.reset_index(drop = True)
	test_e = emoji_test.reset_index(drop = True)
	print (KNN_model(train_e,test_e, y1, y2))
	print (SVMs_model(train_e, test_e, y1, y2))
	print (LR_model(train_e, test_e, y1, y2))

	#Bag of Words
	print ('BW FEATURIZER')
	vec = CountVectorizer(analyzer = 'word', max_features = 6000, min_df = 2)
	train_f = vec.fit_transform(train_b['STATUS'])
	train_f = (pd.DataFrame(train_f.toarray(), columns = vec.get_feature_names()))
	test_f = vec.transform(test_b['STATUS'])
	test_f = (pd.DataFrame(test_f.toarray(), columns = vec.get_feature_names()))
	print (KNN_model(train_f,test_f, y1, y2))
	print (SVMs_model(train_f, test_f, y1, y2))
	print (LR_model(train_f, test_f, y1, y2))

	#LIWC features
	print ('LIWC FEATURIZER')
	dicty_train = dictff(train_l)
	train_l = dicty_train.reset_index(drop = True)
	dicty_test= dictff(test_l)
	test_l = dicty_test.reset_index(drop = True)
	print (KNN_model(dicty_train, dicty_test, y1, y2))
	print (SVMs_model(dicty_train, dicty_test, y1, y2))
	print (LR_model(dicty_train, dicty_test, y1, y2))

	#emoji featurizer+LIWC
	print ('EMODICT')
	emodict_train = pd.concat([dicty_train, emoji_train], axis = 1)
	emodict_test = pd.concat([dicty_test, emoji_test], axis = 1)
	print (KNN_model(emodict_train,emodict_test, y1, y2))
	print (SVMs_model(emodict_train, emodict_test, y1, y2))
	print (LR_model(emodict_train, emodict_test, y1, y2))

	#emoji vectorizer + Bag of Words
	print ('EMO+BW')
	emoCV_train = pd.concat([train_f,train_e], axis = 1)
	emoCV_test = pd.concat([test_f, test_e], axis = 1)
	print (KNN_model(emoCV_train,emoCV_test, y1, y2))
	print (SVMs_model(emoCV_train, emoCV_test, y1, y2))
	print (LR_model(emoCV_train, emoCV_test, y1, y2))

	#LIWC + BW
	print ('LIWC+BW')
	dictCV_train = pd.concat([train_l,train_f], axis = 1)
	dictCV_test = pd.concat([test_l, test_f], axis = 1)
	print (KNN_model(dictCV_train,dictCV_test, y1, y2))
	print (SVMs_model(dictCV_train, dictCV_test, y1, y2))
	print (LR_model(dictCV_train, dictCV_test, y1, y2))

	#emojifeaturizer+dictionary+CV
	print ('TRES')
	tres_train = pd.concat([dictCV_train,train_e], axis = 1)
	tres_test = pd.concat([dictCV_test, test_e], axis = 1)
	print (KNN_model(tres_train, tres_test, y1, y2))
	print (SVMs_model(tres_train, tres_test, y1, y2))
	print (LR_model(tres_train, tres_test, y1, y2))