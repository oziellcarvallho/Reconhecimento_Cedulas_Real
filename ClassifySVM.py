from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.svm import SVC
from pickle import load, dump
from numpy import mean
from os import path
import time
from glob import glob
from sklearn.model_selection import GridSearchCV

n_test = ['1', '2', '3', '4', '5']

def train(x_treino, y_treino, x_teste, filename):
    #params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [0.1, 1, 100, 1000]}
    classifier = SVC(C= 1000, kernel = 'poly')
    classifier.fit(x_treino, y_treino)
    pred = classifier.predict(x_teste)
    dump(classifier, open(filename, 'wb'))
    return pred

def pickle_load(file_name):
    with open(file_name,'rb') as infile:
        data = load(infile)
        infile.close()
    return data

def load_data(path_pickles):
    aux_split = path_pickles.split('/')
    path_arq = './Tests/'+ aux_split[2] +'/models/'+ aux_split[4] +'/'
    filename = path_arq + 'ModelSVC_'+ aux_split[-2] +'_SVM.sav'
    
    print(path_pickles)
    print(path_arq)
    print(filename)

    x_treino = pickle_load(path_pickles + 'Data_Treino.pickle')
    y_treino = pickle_load(path_pickles + 'Label_Treino.pickle')
    x_teste = pickle_load(path_pickles + 'Data_Teste.pickle')
    y_teste = pickle_load(path_pickles + 'Label_Teste.pickle')

    print(x_treino.shape)
    print(y_treino.shape)
    print(x_teste.shape)
    print(y_teste.shape)

    print('SVC 🟢')
    t0 = time.time()
    pred = train(x_treino, y_treino, x_teste, filename)
    t1 = time.time()
    
    description = []
    description.append('Pickle: {0}'.format(aux_split[-2]))
    description.append('Tempo Total: {0}'.format(t1 - t0))
    description.append('Matriz de Confusão:\n{0}'.format(confusion_matrix(y_teste, pred)))
    description.append('Acurácia: {0}'.format(accuracy_score(y_teste, pred)))
    description.append('Kappa: {0}'.format(cohen_kappa_score(y_teste, pred)))
    print("\n".join(description))
    file = open(path_arq + 'Description.txt', 'a')
    file.write('\nSVC')
    file.write('\n' +"\n".join(description) + '\n')
    file.close()

for test in n_test:
    for alg in glob('./Tests/Test'+ test + '/pickles/*'):
        for path_pickles in glob(alg + '/*/'):
            load_data(path_pickles)
