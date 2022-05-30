import numpy as np
from sklearn.model_selection import train_test_split

def stdLimiter(data,limiter):
    colun = []
    for i in data.columns[:-1]:
        if(np.std(data[i].values)/np.mean(data[i].values)>limiter):
            colun.append(i)
    colun = np.array(colun)
    return colun

def Class(values,labels):
    for i in range(len(labels)):
        result = np.where(values == labels[i])
        values[result] = i
    return values

def dataTransform(data):
    X = data[stdLimiter(data,-1)].values
    X = ((X - np.mean(X.T,axis = 1))/np.std(X.T,axis = 1)) #Normalização
    Y = data['Class']
    labels = np.copy(Y.unique())
    return X,data['Class'].values,labels


def TreinoTest(X,Y,test_size,seed,labels):
    X, x, Y, y = train_test_split(X, Y, test_size=test_size, random_state=seed)
    R_treino = np.concatenate((X,Class(np.copy(Y),labels).reshape(-1,1)),axis = 1).astype(float)
    R_teste = np.concatenate((x,Class(np.copy(y),labels).reshape(-1,1)),axis = 1).astype(float)
    return X,x,Y,y,R_treino,R_teste
