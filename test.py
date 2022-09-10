from Model.model import stacking, models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from Data.Process import process_data, shift_name
import joblib
from sklearn.preprocessing import normalize

#Y_test = np.loadtxt("Data/Processed/Y_train.txt")
model = joblib.load('regression.pkl')

X_test = np.loadtxt("Data/Processed/SiAl_demo.dump")
#result = np.loadtxt("Data/Processed/Y_train.txt")
a, b = model.test(X_test)
a = []
for i in range(b.shape[0]):
    a.append(np.argmax(b[i, :]))
b[b<0] = 0
b = normalize(b, axis=1, norm='l1')
np.savetxt("class.txt", b)
data = process_data("SiAl_demo.dump").drop(["element"], axis=1)
r = pd.read_csv("class.txt", names=['BCC','CaF2','CsCl','Dmd','FCC','HCP','Liquid','NaCl'],delimiter=' ')
#print(Fe)
re = pd.concat([data, r], axis = 1)
print(r)
print(re)
np.savetxt("classification.txt", re.fillna(0).values)