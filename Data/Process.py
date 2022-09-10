import pandas as pd
import numpy as np
import pyscal as pc
from sklearn.preprocessing import OneHotEncoder

def ohe(y):
    element = np.unique(y.values).tolist()
    data = pd.DataFrame(columns = element)
    enc = OneHotEncoder(categories = [element]).fit([[el] for el in element])
    array = enc.transform(y.values.reshape(-1,1)).toarray()
    for i in range(len(element)):
        data[element[i]] = array[:,i]
    return data

def shift_name(df, parameter = 2):
    names = df.columns[parameter:]
    df = df.iloc[:,:-parameter]
    df.columns = names
    return df


def process_data(file_name, parameters_row=np.linspace(0, 7), train = False, number_row=[2, 3], boundary_row=[4, 5, 6, 7]):
    # qvals-start
    name = "Data/Raw/" + file_name
    df = shift_name(pd.read_csv(name, delimiter = r"\s+", skiprows = lambda x: x in parameters_row))
    sys = pc.System()
    sys.read_inputfile(name)
    atoms = []
    for index in df.index:
        p = df.loc[index,'x':'z'].values.astype(float).tolist()
        tempAtom = pc.Atom(pos=p, id=int(df['id'].loc[index]), type=int(df['type'].loc[index]))
        atoms.append(tempAtom)
    sys.atoms = atoms

    sys.find_neighbors(method='cutoff', cutoff='adaptive', filter='type')
    sys.calculate_q([4, 6, 8, 10, 12])
    qvals = sys.get_qvals([4, 6, 8, 10, 12])
    i = 2
    for q in qvals:
        i = i + 2
        df = pd.concat([df, pd.Series(q).rename("qval_{}".format(i))], axis=1)

    sys.find_neighbors(method='cutoff', cutoff='adaptive', filter='type_r')
    sys.calculate_q([4, 6, 8, 10, 12])
    qvals = sys.get_qvals([4, 6, 8, 10, 12])
    i = 2
    for q in qvals:
        i = i + 2
        df = pd.concat([df, pd.Series(q).rename("qval'_{}".format(i))], axis=1)
    np.savetxt("Data/Processed/"+file_name, df.loc[:,"qval_4":"qval'_12"].fillna(0).values)
    if train:
        np.savetxt("Data/Processed/"+"y_"+file_name, ohe(df['element']).fillna(0).values)
    print(df)
    return df

#process_data("ZnS.dump", train = True)
process_data("ball_demo.dump")
#process_data("FeAlSiMg.dump")
#process_data("SiAl.dump")