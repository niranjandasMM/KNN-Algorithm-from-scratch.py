import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from collections import defaultdict
from math import sqrt
from collections import Counter
from sklearn import datasets

#Datset in my github repo , use the specifies dataset , either its goona mess up  
dia = pd.read_csv('/content/pima-diabetes-numeric.csv') # you can download the datset from my kaggle dataset repo ... from github |

dia = dia[['preg','plas','pres','skin','insu','mass','pedi','age','class']]
dia = dia.values.tolist()
data = defaultdict(list)

for i in dia:
  data[i[-1]].append(i[:-1]) 
  
def knn(predict,data,k=3):
  distance = []
  for group in data:
    for features in data[group]:
      eucd_dist = sqrt( (features[0]-predict[0])**2 + (features[1]-predict[1])**2 )
      distance.append([eucd_dist,group])
  
  votes = [i[1] for i in sorted(distance)[:5]]
  print(f"the votes are : {votes}")

  vote_result = Counter(votes).most_common()[0][0]
  return vote_result

y_pred =  [6,148,72,35,0,33.6,0.627,50] # = This inputs(features) Belongs to class(Target) 1 
knn(y_pred,data) 
