from Model.model import stacking, models
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

X_train = np.loadtxt("Data/Processed/x_train.txt")
Y_train = np.loadtxt("Data/Processed/Y_train.txt")
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.1)
model = models(Y_train.shape[1])
model.fit(x_train, y_train)
joblib.dump(model, "regression_1.pkl")
#print("loaded")
#print(model.accuracy(X_train, Y_train))