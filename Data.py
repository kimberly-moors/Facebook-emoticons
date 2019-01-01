import csv
import pandas as pd
import sklearn as sk
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from dictfeaturizer import *
from emojifeaturizer import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

	file= open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv", encoding='latin-1')
	dataset= pd.read_csv(file)
	X = dataset.loc[:, ['STATUS']]
	y= dataset.cEXT	
	le = LabelEncoder()
	y= le.fit_transform(dataset['cEXT'])		

	#Countvecorizer
	vec=CountVectorizer() 
	Z=vec.fit_transform(dataset['STATUS'])  															
	features= (pd.DataFrame(Z.toarray(), columns=vec.get_feature_names()))

	#LIWC features
	file= open ("D:\Thesis\mypersonality_final\LIWC_English.csv",encoding='latin-1')
	dictionary = csv.DictReader(file, delimiter=',')
	dict_list = []
	for line in dictionary:
		dict_list.append(line)
	itty = {}	
	i=0	
	for item in dict_list:
		key= dict_list[i].values()
		value=dict_list[i].values()
		dictionaryX = {key, value}
		i+=1
		for item in dictionaryX:
			lils = (list(item))
			itty[(lils[0])] = lils[1]
	dictf= DictFeaturizer(dictionary= itty)		

	#emoji features
	emoji= emoji_featurizer(X)


	#train-test-split
	X1,X2,y1,y2=train_test_split(emoji,y, random_state=1, train_size=0.9, test_size=0.1)
	trainingset= X1
	testset= X2
	
	#KNN
	model=KNeighborsClassifier(n_neighbors=4)
	model.fit(trainingset,y1)
	ss= cross_val_score(model, trainingset, y1, cv=10)
	pp=model.predict(testset)
	print(accuracy_score(y2,pp))

	#SVM
	model_2= svm.SVC(gamma='scale')
	model_2.fit(trainingset,y1)
	tt=cross_val_score(model_2,trainingset,y1,cv=10)
	pp=model_2.predict(testset)
	print(accuracy_score(y2,pp))

	#Logistic Regression
	model_3=LogisticRegression(random_state=0, solver='lbfgs')
	model_3.fit(trainingset,y1)
	tt=cross_val_score(model_3, trainingset,y1,cv=10)
	pp=model_3.predict(testset)
	print(accuracy_score(y2,pp))
