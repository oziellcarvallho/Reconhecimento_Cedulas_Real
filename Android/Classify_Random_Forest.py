from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from cv2 import ml, TERM_CRITERIA_MAX_ITER
from pickle import load
from numpy import mean

path = '../Tests/Test1/pickles/MSER/Surf_MSERNome Padrão:Feature2D.MSER_Delta:2_MaxArea:14400_MinArea:60_Pass2Only:False_Descriptor Size:0-_Descriptor_Size:64_HessianThreshold:200.0_NOctaveLayers:21_NOctaves:22_mean/'

def accuracy(y_true, y_pred, normalize=True):
    accuracy=[]
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    if normalize==True:
        return mean(accuracy)
    if normalize==False:
        return sum(accuracy)

def train(samples, flags, x_teste):
    print('Random Forest Classifier 🟢')
    classifier = ml.RTrees_create()
    
    classifier.setMaxDepth(17)
    classifier.setMaxCategories(6)
    classifier.setCalculateVarImportance(True)
    classifier.setMinSampleCount(0)
    
    term_type, n_trees, forest_accuracy = TERM_CRITERIA_MAX_ITER, 300, 1
    classifier.setTermCriteria((term_type, n_trees, forest_accuracy))
    
    train_data = ml.TrainData_create(samples=samples, layout=ml.ROW_SAMPLE, responses=flags)
    classifier.train(train_data)
    
    classifier.save('./Model_Andorid.sav')
    _ret, responses = classifier.predict(x_teste)
    return responses.ravel()

def pickle_load(filename):
    with open(filename,'rb') as infile:
        data = load(infile)
        infile.close()
    return data

def load_data():    
    x_treino = pickle_load(path + 'Data_Treino.pickle')
    y_treino = pickle_load(path + 'Label_Treino.pickle')
    x_teste = pickle_load(path + 'Data_Teste.pickle')
    y_teste = pickle_load(path + 'Label_Teste.pickle')

    print(x_treino.shape)
    print(y_treino.shape)
    print(x_teste.shape)
    print(y_teste.shape)

    pred = train(x_treino, y_treino, x_teste)
    acc = accuracy(y_teste, pred)
    
    print('Matriz de Confusão:\n', confusion_matrix(y_teste, pred))
    print('Acurácia: ', accuracy_score(y_teste, pred), '  🟢  ', acc)
    print('Kappa: ', cohen_kappa_score(y_teste, pred))
    print('Precisão: ', precision_score(y_teste, pred, average = None))
    print('Sensibilidade: ', recall_score(y_teste, pred, average = None))

load_data()