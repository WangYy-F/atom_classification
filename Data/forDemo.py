import pandas as pd
import numpy as np
from Process import process_data, shift_name

Fe = process_data("C.dump")
r = pd.read_csv("c.txt")
print(Fe)
re = pd.concat([Fe.drop(columns = "element", inplace=True), r], axis = 1)
print(re)
np.savetxt("C_re.txt", re.fillna(0).values)

#FeAlSiMg = process_data("FeAlSiMg.dump")
#SiAl = process_data("SiAl.dump")