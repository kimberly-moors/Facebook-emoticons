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

	#feature pipeline
	estimators=[('vec', CountVectorizer()), ('dict', DictFeaturizer(trainingset))]

	
	#emoji features
	def emoji_featurizer (dataset):
		#the emoji's placed into a dictionary.
		emojidictionary={}
		emojidictionary['happy']= [':)',':]','XD',':-)','^_^',':D',':P','xD',':}',':>', '=)']
		emojidictionary['random']= ['<3',':$',':o']
		emojidictionary['winking']= [';)',';-)']
		emojidictionary['neutral']= [':|','-.-']
		emojidictionary['sad']= [':(',':[',':-(','=(',':{','://',":\'(",':c']
		emojidictionary['angry']= ['^^',':@',':S','>:(','-_-']
		emojis = []
		lines=[]
		yoepi=[]

		for value in emojidictionary.values():
			for v in value:
				emojis.append(v)
		for line in dataset['STATUS']:
			words=line.split(" ")
			for word in words:
				yoepi.append(word)
				for value in emojidictionary.values():
					for x in value:
						if x in yoepi:
							lines.append(line)
		return(lines)
	

	emoji= emoji_featurizer(trainingset)
	pip=FeatureUnion(estimators)
	pipy = pip.fit(trainingset, y1)
