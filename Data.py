import csv
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


dataset= open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv")
dataset= pd.read_csv(dataset)
X=dataset.loc[:,['STATUS', 'cEXT']]
y = dataset.index

for i in X: 																					#labeling 0 = introvert, 1= extravert
	if X.cEXT == 'y':
	 	print(row)

#Which smileys appear in my data?															
j= 0														
for string in X['STATUS']:									
	i=0															
	if ':@' in string: 																		
		i=+1											
		j+=1 																				
		print(string)
print(j)


#Count Vectorizer features
vec=CountVectorizer()
Z=vec.fit_transform(X['STATUS'])    															#probleem = geen utf-code
features= (pd.DataFrame(Z.toarray(),columns=vec.get_feature_names()))


#Splitting the data in a trainingset and holdout data
X1,X2,y1,y2=train_test_split(X,y, random_state=1, train_size=0.9)
trainingset= X1
testset= X2


