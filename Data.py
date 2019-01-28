import csv
import pandas as pd
import sklearn as sk
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from dictfeaturizer import *
from pfffff import *
from emojifeaturizer import *

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, learning_curve, GridSearchCV
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, mean_absolute_error, classification_report, f1_score
from emoji_gone import *

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
   		X1,X2,y1,y2 = train_test_split(featureset, y, random_state = 1, train_size = 0.9, test_size = 0.1)
   		trainingset = X1
   		testset = X2
   		return (trainingset, testset, y1, y2)
  
  	#KNN
	def KNN_model(featureset, y):
		trainingset, testset, y1, y2 = train_test_features(featureset, y)
		model = KNeighborsClassifier(n_neighbors = 2)
		model.fit (trainingset,y1)
		ss = cross_val_score(model,trainingset,y1, cv = 10)
		pp = model.predict(testset)
		acc = accuracy_score(y2,pp)
		conf = confusion_matrix(y2,pp)
		prec = precision_score(y2,pp)
		rec = recall_score(y2,pp)
		roc = roc_auc_score(y2,pp)
		MAE = mean_absolute_error(y2,pp)
		F1 = f1_score(y2,pp)
		return('KNN results:', 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
 
	#SVMs
	def SVMs_model(featureset,y):
		trainingset, testset, y1, y2 = train_test_features(featureset, y)
		model= svm.SVC(C=10, gamma='scale', kernel='rbf')
		model.fit(trainingset,y1)
		ss = cross_val_score(model,trainingset,y1, cv = 10)
		pp = model.predict(testset)
		acc = accuracy_score(y2,pp)
		conf = confusion_matrix(y2,pp)
		prec = precision_score(y2,pp)
		rec = recall_score(y2,pp)
		roc = roc_auc_score(y2,pp)
		MAE = mean_absolute_error(y2,pp)
		F1 = f1_score(y2,pp)
		return('SVMs results:', 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
 
   #Logistic Regression
	def LR_model(featureset, y):
		trainingset, testset, y1, y2 = train_test_features(featureset, y)		
		model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=300)
		model.fit(trainingset,y1)
		ss = cross_val_score(model,trainingset,y1, cv = 10)
		pp = model.predict(testset)
		acc = accuracy_score(y2,pp)
		conf = confusion_matrix(y2,pp)
		prec = precision_score(y2,pp)
		rec = recall_score(y2,pp)
		roc = roc_auc_score(y2,pp)
		MAE = mean_absolute_error(y2,pp)
		F1 = f1_score(y2,pp)
		return('LR results:', 'crossvalidation =', ss, 'accuracy =', acc, 'confusion matrix =', conf, 'precision =', prec, 'recall =', rec, "roc =", roc, 'MAE =', MAE, 'F1 =', F1 )
 
  	
  	#emoji features
	print ('EMOJI FEATURIZER')
	emoji = emoji_featurizer(D)
	print (KNN_model(emoji,y))
	print (SVMs_model(emoji, y))
	print (LR_model(emoji, y))
 	
 	 #Countvecorizer
	print ('CV FEATURIZER')
	vec = CountVectorizer() 
	Z = vec.fit_transform(dataset['STATUS'])  															
	features = (pd.DataFrame(Z.toarray(), columns=vec.get_feature_names()))
	print (KNN_model(features, y))
	print (SVMs_model(features, y))
	print (LR_model(features, y))
  
	#LIWClike features
	print ('DICT FEATURIZER')
	dicty = dictff(X,itty)
	print (KNN_model(dicty, y))
	print (SVMs_model(dicty, y))
	print (LR_model(dicty, y))
 	
	#emoji featurizer+dictionary
	print ('EMODICT')
	emodict = pd.concat([dicty, emoji], axis=1)
	print (KNN_model(emodict, y))
	print (SVMs_model(emodict, y))
	print (LR_model(emodict, y))
 	
	#emoji vectorizer + CV
	print ('EMOCV')
	emoCV = pd.concat([features,emoji], axis=1)
	print (KNN_model(emoCV, y))
	print (SVMs_model(emoCV, y))	
	print (LR_model(emoCV, y))
 	
	#dictionary + CV
	print ('DICTCV')
	dictCV = pd.concat([dicty,features], axis=1)
	print (KNN_model(dictCV, y))
	print (SVMs_model(dictCV, y))
	print (LR_model(dictCV, y))
 	
	"""#emojifeaturizer+dictionary+CV
	tres=pd.concat([dictCV,emoji], axis=1)
	X1,X2,y13,y14=train_test_split(tres,y,random_state=1, train_size=0.9, test_size=0.1)
	trainingset_7=X1
	testset_7=X2
	
	

	

	

