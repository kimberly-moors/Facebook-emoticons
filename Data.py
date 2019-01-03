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
from pfffff import *
from emojifeaturizer import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

	file= open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv", encoding='latin-1')
	dataset= pd.read_csv(file)
	X = dataset.loc[:, ['STATUS']]
	D=	dataset.loc[:, ['STATUS']]
	y= dataset.cEXT	
	le = LabelEncoder()
	y= le.fit_transform(dataset['cEXT'])		


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

	#Countvecorizer
	vec=CountVectorizer() 
	Z=vec.fit_transform(dataset['STATUS'])  															
	features= (pd.DataFrame(Z.toarray(), columns=vec.get_feature_names()))

	#LIWClike features
	dicty= dictff(X,itty)
	
	#emoji features
	emoji= emoji_featurizer(D)


	#train-test-split
	X1,X2,y1,y2=train_test_split(emoji,y, random_state=1, train_size=0.9, test_size=0.1)
	trainingset= X1
	testset= X2

	X1,X2,y3,y4=train_test_split(features,y, random_state=1, train_size=0.9, test_size=0.1)
	trainingset_2= X1
	testset_2= X2

	X1,X2,y5,y6=train_test_split(dicty,y,random_state=1, train_size=0.9, test_size=0.1)
	trainingset_3=X1
	testset_3=X2

	
	
	#KNN
	model=KNeighborsClassifier(n_neighbors=4)
	model.fit(trainingset,y1)
	pp=model.predict(testset)
	ss= cross_val_score(model, trainingset, y1, cv=10)
	print(ss)
	print(accuracy_score(y2,pp))

	model.fit(trainingset_2,y3)
	pp=model.predict(testset)
	cs=cross_val_score(model,trainingset_2,y3, cv=10)
	print(cs)
	print(accuracy_score(y4,pp))

	fun= model.fit(trainingset_3,y5)
	pp=fun.predict(testset_3)
	qq=cross_val_score(fun,trainingset_3,y5,cv=10)
	print(qq)
	print(accuracy_score(y6,pp))


	#SVM
	model_2= svm.SVC(gamma='scale')
	model_2.fit(trainingset,y1)
	pp=model_2.predict(testset_2)
	tt=cross_val_score(model_2,trainingset,y1,cv=10)
	print(tt)
	print(accuracy_score(y2,pp))

	model_2.fit(trainingset_2,y3)
	pp= model_2.predict(testset_2)
	tt=cross_val_score(model_2, trainingset_2,y3, cv=10)
	print(tt)
	print(accuracy_score(y4,pp))

	model_2.fit(trainingset_3,y5)
	pp= model_2.predict(testset_3)
	tt=cross_val_score(model_2, trainingset_3,y5, cv=10)
	print(tt)
	print(accuracy_score(y6,pp))


	#Logistic Regression
	model_3=LogisticRegression(random_state=0, solver='lbfgs', max_iter=300)
	model_3.fit(trainingset,y1)
	pp=model_3.predict(testset)
	tt=cross_val_score(model_3, trainingset,y1,cv=10)
	print(tt)
	print(accuracy_score(y2,pp))

	model_3.fit(trainingset_2,y3)
	pp=model_3.predict(testset_2)
	tt=cross_val_score(model_3, trainingset_2,y3,cv=10)
	print(tt)
	print(accuracy_score(y4,pp))

	model_3.fit(trainingset_3,y5)
	pp=model_3.predict(testset_3)
	tt=cross_val_score(model_3, trainingset_3,y5,cv=10)
	print(tt)
	print(accuracy_score(y6,pp))