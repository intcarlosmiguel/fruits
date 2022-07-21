# Bibliotecas de Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Bibliotecas Auxiliares
import numpy as np
import pickle
from os.path import exists
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#Biblioteca de Validação
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

#Biblioteca Pytorch
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

def CM(y_true,y_pred,labels,file):
    cf_matrix = confusion_matrix(y_true, y_pred)
    acc = np.trace(cf_matrix)/np.sum(cf_matrix)
    print(f'Acurácia desse modelo é de: {acc}.')
    plt.figure(figsize = (11.7,8.27))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('Matriz de Confusão\n\n');
    ax.set_xlabel('\nValores Preditos')
    ax.set_ylabel('Valores Verdadeiros ');
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.savefig(file, dpi=300)
    plt.show()
    return acc

def doubleClasfier(pred_1,pred_2,acc1,acc2):
    pred = []
    for i in range(len(pred_1)):
        if(pred_1[i]==pred_2[i]):
            pred.append(pred_1[i])
        else:
            if(acc1>acc2):
                pred.append(pred_1[i])
            else:
                pred.append(pred_2[i])
    return np.array(pred)

def createClassifier(X,Y):
    print('Carregando modelos..........')
    if(not exists('neural.sav')):
        print('Carregando Rede Neural..........')
        parametros = {
            'activation': ["identity", 'logistic', 'tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'tol' : [1e-4],
            'max_iter' : [5000],
            'hidden_layer_sizes' : [(i,i) for i in range(30,41)]
        }
        bestClassfier(X,Y,parametros,MLPClassifier,'./models/neural.sav')
    if(not exists('SVM.sav')):
        print('Carregando SVM..........')
        parametros = {
            'kernel': ['rbf','linear','poly','sigmoid'],
            'C': [(i/10)for i in range(1,41)],
            'tol': [(1/10**i)for i in range(6)]
        }
        bestClassfier(X,Y,parametros,SVC,'./models/SVM.sav')
    if(not exists('Randomforest.sav')):
        print('Carregando Random Forest..........')
        parametros = {
            'criterion': ['gini','entropy'],
            'n_estimators': [(i) for i in range(10,100,10)],
            'min_samples_split': [(i)for i in range(4,12,2)],
            'min_samples_leaf': [(i)for i in range(3,12,2)]
        }
        bestClassfier(X,Y,parametros,RandomForestClassifier,'./models/Randomforest.sav')
    if(not exists('Logistic.sav')):
        print('Carregando Regressão Logística..........')
        parametros = {
            'penalty': ['l2'],
            'C': [(i/10) for i in range(20,41)],
            'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter' : [7000]
        }
        bestClassfier(X,Y,parametros,LogisticRegression,'./models/Logistic.sav')
    if(not exists('neighbors.sav')):
        print('Carregando KNN..........')
        parametros = {
            'weights': ['uniform', 'distance'],
            'n_neighbors': [(int(i)) for i in range(1,11)],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p' : [1,2]
        }
        bestClassfier(X,Y,parametros,KNeighborsClassifier,'./models/neighbors.sav')
    print('Modelos Carregados!')
    return pickle.load(open('./models/neural.sav','rb')),pickle.load(open('./models/SVM.sav','rb')),pickle.load(open('./models/Randomforest.sav','rb')),pickle.load(open('./models/Logistic.sav','rb')),pickle.load(open('./models/neighbors.sav','rb'))

def bestClassfier(X,Y,parameters,classfier,namefile):
    grid = GridSearchCV(estimator = classfier(),param_grid=parameters)
    grid.fit(X,Y)
    best_pa = grid.best_params_
    best = grid.best_score_
    print(best_pa,best)
    pickle.dump(classfier(**best_pa),open(namefile,'wb'))

def Classfier(classfier,X,Y,x,y,labels,file):
    classfier.fit(X,Y)
    previsoes = classfier.predict(x)
    acc = accuracy_score(y,previsoes)
    CM(y,previsoes,labels,file)
    return acc, previsoes

def TorchNN(R_treino,R_teste,epocas,labels,file):
    treino_tamanho = int(R_treino.shape[0])
    teste_tamanho = int(R_teste.shape[0])
    R_treino = torch.tensor(R_treino)
    R_teste = torch.tensor(R_teste)
    dimensao_mnist = 34
    
    modelo_ = nn.Sequential(
        nn.Linear(dimensao_mnist, 128),
        nn.ReLU(), # funcao relu
        nn.Linear(128,128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    otimizador = optim.SGD(modelo_.parameters(), lr=1e-2, momentum=0.9)
    loader_treino = DataLoader(R_treino, batch_size=treino_tamanho)
    loader_validacao = DataLoader(R_teste, batch_size=teste_tamanho)
    funcao_erro = nn.CrossEntropyLoss()
    
    # Passa por todos os datapoints de treino
    for epoca in range(epocas):
        acertos_treino = 0
        acertos_validacao = 0

        for batch in loader_treino:
            x = batch[:,:-1]
            y_real = torch.transpose(batch,0,1)[-1]
            b = x.size(0)

            y_previsto = modelo_(x.float())
            erro = funcao_erro(y_previsto, y_real.type(torch.LongTensor))
            predicoes = torch.max(y_previsto.data, 1)[1]

            acertos_batch = (predicoes == y_real).sum()
            acertos_treino += acertos_batch
            
            modelo_.zero_grad()
            erro.backward()
            otimizador.step()

        with torch.no_grad():
            for batch in loader_validacao:
                x = batch[:,:-1]
                y_real = torch.transpose(batch,0,1)[-1]

                b = x.size(0)
                x = x.view(b, -1)

                y_previsto = modelo_(x.float())

                predicoes = torch.max(y_previsto.data, 1)[1]
                acertos_batch = (predicoes == y_real).sum()
                acertos_validacao += acertos_batch
        
        err = erro.item()
        acc_ = acertos_treino/treino_tamanho
        acc = acertos_validacao/teste_tamanho
        #print(f'Ep: {epoca + 1}, Erro treino: {err:.5f}, Acc treino: {acc_:.5f}, Acc valid: {acc:.5f}')
    model_scripted = torch.jit.script(modelo_)
    model_scripted.save('./img/NNtorch.pt')
    predicoes = predicoes.numpy()
    y_real = y_real.numpy()
    acc = CM(y_real,predicoes,labels,file)
    return acc,predicoes