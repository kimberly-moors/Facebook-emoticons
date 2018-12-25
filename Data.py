import csv
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline
from dictfeaturizer import *
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

	file= open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv", encoding='latin-1')
	dataset= pd.read_csv(file)
	X = dataset.loc[:, ['STATUS']]
	y= dataset.cEXT	
	le = LabelEncoder()
	y= le.fit_transform(dataset['cEXT'])		

	#train-test-split
	X1,X2,y1,y2=train_test_split(X,y, random_state=1, train_size=0.9, test_size=0.1)
	trainingset= X1
	testset= X2

	#Countvecorizer
	vec=CountVectorizer() 
	Z=vec.fit_transform(trainingset['STATUS'])  															
	features= (pd.DataFrame(Z.toarray(), columns=vec.get_feature_names()))

	#LIWC features
	dictf= DictFeaturizer(dictionary= trainingset)		

	#feature pipeline
	estimators=[('vec', CountVectorizer()), ('dict', DictFeaturizer(trainingset))]
	pip=FeatureUnion(estimators)
	pipy = pip.fit(trainingset, y1)
