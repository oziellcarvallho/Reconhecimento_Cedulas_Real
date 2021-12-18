from numpy import array
from pickle import load
from glob import glob
from statistics import median
from cv2 import imread, xfeatures2d
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

surf = xfeatures2d.SURF_create(hessianThreshold = 200, nOctaves = 22, nOctaveLayers = 21)
print('Tamanho do Descritor: ', surf.descriptorSize())
print('Extendido? ', surf.getExtended())
print('HessianThreshold: ', surf.getHessianThreshold())
print('Upright: ', surf.getUpright())
print('NOctaveLayers: ', surf.getNOctaveLayers())
print('NOctaves: ', surf.getNOctaves())

filename = './Tests/Test1/pickles/MSER/Surf_MSERNome Padrão:Feature2D.MSER_Delta:2_MaxArea:14400_MinArea:60_Pass2Only:False_Descriptor Size:0-_Descriptor_Size:64_HessianThreshold:200.0_NOctaveLayers:21_NOctaves:22_mean_RandomForest.sav'
loaded_model = load(open(filename, 'rb'))

def teste(path):
    print(path)
    img = imread(path, 0)
    kp, des = surf.detectAndCompute(img, None)
    dado = []
    a = []
    for j in range(64):
        a.append(des.transpose()[j].mean())
    dado.append(a)
    dados = array(dado)
    result = loaded_model.predict(dados)
    print(result[0])
    return result[0]

def run(path):
    x = []
    y = []
    classes = ['2', '5', '10', '20', '100']
    for classe in classes:
        data = glob(path + classe +'/*.jpg')
        for file_path in data:
            a = teste(file_path)
            x.append(a)
            y.append(int(classe))
    resultados = array(x)
    rotulos = array(y)
    return rotulos, resultados

rotulos, resultados = run('./Mini/')
print('Matriz de Confusão:\n', confusion_matrix(rotulos, resultados))
print('Acurácia: ', accuracy_score(rotulos, resultados))
print('Kappa: ', cohen_kappa_score(rotulos, resultados))