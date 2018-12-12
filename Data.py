import csv
import pandas as pd
import sklearn as sk
import numpy as np
dataset= open ("D:\Thesis\mypersonality_final\mypersonality_final2.csv")
dataset= pd.read_csv(dataset)
X=dataset.loc[:,['STATUS', 'cEXT', 'sEXT']]
y = dataset.index
