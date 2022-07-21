# Importação das Bibliotecas e outros arquivos.
import pandas as pd
from transforms import *
from ML import createClassifier,Classfier,doubleClasfier,CM,TorchNN

#Lendo arquivo csv
data = pd.read_excel('Date_Fruit_Datasets.xlsx')
X,Y,labels = dataTransform(data)
X, x, Y, y,R_treino,R_teste = TreinoTest(X, Y, 0.2,42,labels)
#Carregando Classficadores
neural, SVM,forest,logistic,neigh = createClassifier(X,Y)
print('\n')

#Separando dados em teste e treino.


#Modelos de Machine Learning
print('Algoritmo de Rede Neural')
acc_neural,neural_prev = Classfier(neural,X,Y,x,y,labels,'./img/neural.png')
print('=======================================================\n')
print('Algoritmo de Regressão Logística')
acc_logistic, logistic_prev = Classfier(logistic,X,Y,x,y,labels,'./img/Logistic.png')
print('=======================================================\n')
print('Algoritmo de Floresta Aleatória')
acc_forest, forest_prev = Classfier(forest,X,Y,x,y,labels,'./img/Randomforest.png')
print('=======================================================\n')
print('Algoritmo de SVM')
acc_SVM, SVM_prev = Classfier(SVM,X,Y,x,y,labels,'./img/SVM.png')
print('=======================================================\n')
print('Algoritmo de k-vizinhos mais próximos')
acc_neigh,neigh_prev = Classfier(neigh,X,Y,x,y,labels,'./img/neighbors.png')
print('=======================================================\n')

#Junção de Algoritmos
prev = doubleClasfier(neural_prev,neigh_prev,acc_neural,acc_neigh)
print('Combinação de Algoritmos:')
CM(y,prev,labels,'./img/twoalgorithms.png')

# Rede Neural do Pytorch
troch_acc , torch_prev = TorchNN(R_treino,R_teste,500,labels,'./img/NNtorch.png')

prev = doubleClasfier(labels[torch_prev],neigh_prev,troch_acc,acc_neigh)
CM(y,prev,labels,'./img/twoalgorithms2.png')