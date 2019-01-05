import csv
import pandas as pd
import sklearn as sk
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from dictfeaturizer import *
from pfffff import *
from emojifeaturizer import *

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, mean_absolute_error, classification_report, f1_score


if __name__ == "__main__":

	file= open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv", encoding='latin-1')
	dataset= pd.read_csv(file)
	X = dataset.loc[:, ['STATUS']]
	D=	dataset.loc[:, ['STATUS']]
	y= dataset.cEXT	
	le = LabelEncoder()
	y= le.fit_transform(dataset['cEXT'])		


	#convert dataset into dictionary
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
	#dictf= DictFeaturizer(dictionary= itty)		

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
	
	#emoji vectorizer+dictionary
	emodict=pd.concat([dicty, emoji], axis=1)
	X1,X2,y7,y8=train_test_split(emodict,y,random_state=1, train_size=0.9, test_size=0.1)
	trainingset_4=X1
	testset_4=X2

	#eùoji vectorizer + CV
	emoCV=pd.concat([features,emoji], axis=1)
	X1,X2,y9,y10=train_test_split(emoCV,y,random_state=1, train_size=0.9, test_size=0.1)
	trainingset_5=X1
	testset_5=X2

	#dictionary + CV
	dictCV=pd.concat([dicty,features], axis=1)
	dictCV=pd.concat([features,dicty], axis=1)
	X1,X2,y11,y12=train_test_split(dictCV,y,random_state=1, train_size=0.9, test_size=0.1)
	trainingset_6=X1
	testset_6=X2

	#emojifeaturizer+dictionary+CV
	tres=pd.concat([dictCV,emoji], axis=1)
	X1,X2,y13,y14=train_test_split(tres,y,random_state=1, train_size=0.9, test_size=0.1)
	trainingset_7=X1
	testset_7=X2
	
	#KNN
	print('KNN emoji')
	model=KNeighborsClassifier(n_neighbors=4)
	model.fit(trainingset,y1)
	pp=model.predict(testset)
	ss= cross_val_score(model, trainingset, y1, cv=10)
	print(ss)
	print(accuracy_score(y2,pp))
	print(confusion_matrix(y2,pp))
	print(precision_score(y2,pp))
	print(recall_score(y2,pp))
	print(roc_auc_score(y2,pp))
	print(mean_absolute_error(y2,pp))
	print(f1_score(y2,pp))


	print('KNN CV')
	model.fit(trainingset_2,y3)
	pp=model.predict(testset_2)
	ss=cross_val_score(model,trainingset_2,y3, cv=10)
	print(ss)
	print(accuracy_score(y4,pp))
	print(confusion_matrix(y4,pp))
	print(precision_score(y4,pp))
	print(recall_score(y4,pp))
	print(roc_auc_score(y4,pp))
	print(mean_absolute_error(y4,pp))
	print(f1_score(y4,pp))


	print('KNN dicty')
	model.fit(trainingset_3,y5)
	pp=model.predict(testset_3)
	ss=cross_val_score(model,trainingset_3,y5,cv=10)
	print(ss)
	print(accuracy_score(y6,pp))
	print(confusion_matrix(y6,pp))
	print(precision_score(y6,pp))
	print(recall_score(y6,pp))
	print(roc_auc_score(y6,pp))
	print(mean_absolute_error(y6,pp))
	print(f1_score(y6,pp))


	print('KNN EMOJI DICT')
	model.fit(trainingset_4,y7)
	pp=model.predict(testset_4)
	ss=cross_val_score(model,trainingset_4,y7,cv=10)
	print(ss)
	print(accuracy_score(y8,pp))
	print(confusion_matrix(y8,pp))
	print(precision_score(y8,pp))
	print(recall_score(y8,pp))
	print(roc_auc_score(y8,pp))
	print(mean_absolute_error(y8,pp))
	print(f1_score(y8,pp))


	print('KNN emoCV')
	model.fit(trainingset_5,y9)
	pp=model.predict(testset_5)
	ss=cross_val_score(model,trainingset_5,y9,cv=10)
	print(ss)
	print(accuracy_score(y10,pp))
	print(confusion_matrix(y10,pp))
	print(precision_score(y10,pp))
	print(recall_score(y10,pp))
	print(roc_auc_score(y10,pp))
	print(mean_absolute_error(y10,pp))
	print(f1_score(y10,pp))


	print('KNN dictCV')
	model.fit(trainingset_6,y11)
	pp=model.predict(testset_6)
	ss=cross_val_score(model,trainingset_6,y11,cv=10)
	print(ss)
	print(accuracy_score(y12,pp))
	print(confusion_matrix(y12,pp))
	print(precision_score(y12,pp))
	print(recall_score(y12,pp))
	print(roc_auc_score(y12,pp))
	print(mean_absolute_error(y12,pp))
	print(f1_score(y12,pp))


	print('KNN tres')
	model.fit(trainingset_7,y13)
	pp=model.predict(testset_7)
	ss=cross_val_score(model,trainingset_7,y13,cv=10)
	print(ss)
	print(accuracy_score(y14,pp))
	print(confusion_matrix(y14,pp))
	print(precision_score(y14,pp))
	print(recall_score(y14,pp))
	print(roc_auc_score(y14,pp))
	print(mean_absolute_error(y14,pp))
	print(f1_score(y14,pp))
	

	#SVM
	print('SVM emoji')
	model_2= svm.SVC(gamma='scale')
	model_2.fit(trainingset,y1)
	pp=model_2.predict(testset)
	ss=cross_val_score(model_2,trainingset,y1,cv=10)
	print(ss)
	print(accuracy_score(y2,pp))
	print(confusion_matrix(y2,pp))
	print(precision_score(y2,pp))
	print(recall_score(y2,pp))
	print(roc_auc_score(y2,pp))
	print(mean_absolute_error(y2,pp))
	print(f1_score(y2,pp))


	print('SVM CV')
	model_2.fit(trainingset_2,y3)
	pp= model_2.predict(testset_2)
	ss=cross_val_score(model_2, trainingset_2,y3, cv=10)
	print(ss)
	print(accuracy_score(y4,pp))
	print(confusion_matrix(y4,pp))
	print(precision_score(y4,pp))
	print(recall_score(y4,pp))
	print(roc_auc_score(y4,pp))
	print(mean_absolute_error(y4,pp))
	print(f1_score(y4,pp))
	

	print('SVM dicty')
	model_2.fit(trainingset_3,y5)
	pp= model_2.predict(testset_3)
	ss=cross_val_score(model_2, trainingset_3,y5, cv=10)
	print(ss)
	print(accuracy_score(y6,pp))
	print(confusion_matrix(y6,pp))
	print(precision_score(y6,pp))
	print(recall_score(y6,pp))
	print(roc_auc_score(y6,pp))
	print(mean_absolute_error(y6,pp))
	print(f1_score(y6,pp))

	
	print('emodict')
	model_2.fit(trainingset_4,y7)
	pp= model_2.predict(testset_4)
	ss=cross_val_score(model_2, trainingset_4,y7, cv=10)
	print(ss)
	print(accuracy_score(y8,pp))
	print(confusion_matrix(y8,pp))
	print(precision_score(y8,pp))
	print(recall_score(y8,pp))
	print(roc_auc_score(y8,pp))
	print(mean_absolute_error(y8,pp))
	print(f1_score(y8,pp))


	print('SVM emoCV')
	model_2.fit(trainingset_5,y9)
	pp= model_2.predict(testset_5)
	ss=cross_val_score(model_2, trainingset_5,y9, cv=10)
	print(ss)
	print(accuracy_score(y10,pp))
	print(confusion_matrix(y10,pp))
	print(precision_score(y10,pp))
	print(recall_score(y10,pp))
	print(roc_auc_score(y10,pp))
	print(mean_absolute_error(y10,pp))
	print(f1_score(y10,pp))


	print('SVM dictCV')
	model_2.fit(trainingset_6,y11)
	pp= model_2.predict(testset_6)
	ss=cross_val_score(model_2, trainingset_6,y11, cv=10)
	print(ss)
	print(accuracy_score(y12,pp))
	print(confusion_matrix(y12,pp))
	print(precision_score(y12,pp))
	print(recall_score(y12,pp))
	print(roc_auc_score(y12,pp))
	print(mean_absolute_error(y12,pp))
	print(f1_score(y12,pp))
	
	
	print('SVM tres')
	model_2.fit(trainingset_7,y13)
	pp= model_2.predict(testset_7)
	ss=cross_val_score(model_2, trainingset_7,y13, cv=10)
	print(ss)
	print(accuracy_score(y14,pp))
	print(confusion_matrix(y14,pp))
	print(precision_score(y14,pp))
	print(recall_score(y14,pp))
	print(roc_auc_score(y14,pp))
	print(mean_absolute_error(y14,pp))
	print(f1_score(y14,pp))


	#Logistic Regression
	print('LR Emoji')
	model_3=LogisticRegression(random_state=0, solver='lbfgs', max_iter=300)
	model_3.fit(trainingset,y1)
	pp=model_3.predict(testset)
	ss=cross_val_score(model_3, trainingset,y1,cv=10)
	print(ss)
	print(accuracy_score(y2,pp))
	print(confusion_matrix(y2,pp))
	print(precision_score(y2,pp))
	print(recall_score(y2,pp))
	print(roc_auc_score(y2,pp))
	print(mean_absolute_error(y2,pp))
	print(f1_score(y2,pp))

	print('LR CV')
	model_3.fit(trainingset_2,y3)
	pp=model_3.predict(testset_2)
	ss=cross_val_score(model_3, trainingset_2,y3,cv=10)
	print(ss)
	print(accuracy_score(y4,pp))
	print(confusion_matrix(y4,pp))
	print(precision_score(y4,pp))
	print(recall_score(y4,pp))
	print(roc_auc_score(y4,pp))
	print(mean_absolute_error(y4,pp))
	print(f1_score(y4,pp))
	
	print('LR dicty')
	model_3.fit(trainingset_3,y5)
	pp=model_3.predict(testset_3)
	ss=cross_val_score(model_3, trainingset_3,y5,cv=10)
	print(ss)
	print(accuracy_score(y6,pp))
	print(confusion_matrix(y6,pp))
	print(precision_score(y6,pp))
	print(recall_score(y6,pp))
	print(roc_auc_score(y6,pp))
	print(mean_absolute_error(y6,pp))
	print(f1_score(y6,pp))


	print('LR emodict')
	model_3.fit(trainingset_4,y7)
	pp=model_3.predict(testset_4)
	ss=cross_val_score(model_3, trainingset_4,y7,cv=10)
	print(ss)
	print(accuracy_score(y8,pp))
	print(confusion_matrix(y8,pp))
	print(precision_score(y8,pp))
	print(recall_score(y8,pp))
	print(roc_auc_score(y8,pp))
	print(mean_absolute_error(y8,pp))
	print(f1_score(y8,pp))


	print('LR emoCV')
	model_3.fit(trainingset_5,y9)
	pp=model_3.predict(testset_5)
	ss=cross_val_score(model_3, trainingset_5,y9,cv=10)
	print(ss)
	print(accuracy_score(y10,pp))
	print(confusion_matrix(y10,pp))
	print(precision_score(y10,pp))
	print(recall_score(y10,pp))
	print(roc_auc_score(y10,pp))
	print(mean_absolute_error(y10,pp))
	print(f1_score(y10,pp))


	print('LR dictCV')
	model_3.fit(trainingset_6,y11)
	pp=model_3.predict(testset_6)
	ss=cross_val_score(model_3, trainingset_6,y11,cv=10)
	print(ss)
	print(accuracy_score(y12,pp))
	print(confusion_matrix(y12,pp))
	print(precision_score(y12,pp))
	print(recall_score(y12,pp))
	print(roc_auc_score(y12,pp))
	print(mean_absolute_error(y12,pp))
	print(f1_score(y12,pp))


	print('LR tres')
	model_3.fit(trainingset_7,y13)
	pp=model_3.predict(testset_7)
	ss=cross_val_score(model_3, trainingset_7,y13,cv=10)
	print(ss)
	print(accuracy_score(y14,pp))
	print(confusion_matrix(y14,pp))
	print(precision_score(y14,pp))
	print(recall_score(y14,pp))
	print(roc_auc_score(y14,pp))
	print(mean_absolute_error(y14,pp))
	print(f1_score(y14,pp))