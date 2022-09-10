from Model.model import stacking, models
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from Data.Process import process_data

model = joblib.load('finalized_model_6.27_C.sav')
#process_data("ZnS.dump", train = True)
model.extend("Data/Processed/ZnS.dump", "Data/Processed/y_ZnS.dump")
joblib.dump(model, 'finalized_model_6.27_C_ZnS.sav')