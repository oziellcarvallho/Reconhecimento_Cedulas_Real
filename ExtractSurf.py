from cv2 import imread, xfeatures2d
from pickle import dump
from os import mkdir, path
import numpy as np
import time
from statistics import median

n_test = '4'
path_dataset = './Base/'
path_pickles = './Tests/Test' + n_test +'/pickles/SURF/'
dtt = np.load(path_dataset + 'Data_Treino_Teste_'+ n_test +'.npy', allow_pickle=True).item()

def pickle_dump(filename, data):
    with open(filename, 'wb') as pd:
        dump(data, pd)
        pd.close()

def vetor_create(surf, x):
  a = []
  if (x == 'mean'):
    for j in range(surf.descriptorSize()):
      a.append(des.transpose()[j].mean())
  if (x == 'median'):
    for j in range(surf.descriptorSize()):
      a.append(median(des.transpose()[j]))
  print('✔')
  return a


hessianThresholds = [200]
#hessianThresholds = [100, 200, 300]
#nOctavs = [4, 10, 20, 22, 23, 24]
nOctavs = [22]
classes = ['2', '5', '10', '20', '50', '100']
stages = ['Treino', 'Teste']
calculos = ['mean', 'median']
for calc in calculos:
  for ht in hessianThresholds:
      for no in nOctavs:
          description = []
          surf = xfeatures2d.SURF_create(hessianThreshold = ht, nOctaves = no, nOctaveLayers = no - 1)
          description.append('Descriptor_Size:{0}'.format(surf.descriptorSize()))
          description.append('HessianThreshold:{0}'.format(surf.getHessianThreshold()))
          description.append('NOctaveLayers:{0}'.format(surf.getNOctaveLayers()))
          description.append('NOctaves:{0}'.format(surf.getNOctaves()))
          description.append(calc)
          print("\n".join(description))
          path_save = path_pickles + 'SURF' + "_".join(description)
          if not (path.isdir(path_save)):
            mkdir(path_save)
            t0 = time.time()
            for stage in stages:
              x = []
              y = []
              for classe in classes:
                for file_path in dtt[classe][stage]:
                  print('Caminho', file_path)
                  img = imread(file_path, 0)
                  kp, des = surf.detectAndCompute(img, None)
                  print(len(kp))
                  print(len(des))
                  a = vetor_create(surf, calc)
                  x.append(a)
                  y.append(int(classe))
              dados = np.array(x)
              rotulos = np.array(y)
              print('Dados ' + stage, dados)
              print('Rótulos ' + stage, rotulos)
              index = np.random.permutation(len(rotulos))
              X, Y = dados[index], rotulos[index]
              pickle_dump(path_save + '/Data_'+ stage +'.pickle', X)
              pickle_dump(path_save + '/Label_'+ stage +'.pickle', Y)
            t1 = time.time()
            file = open(path_pickles + 'Description.txt', 'a')
            file.write('\nSURF\n' +"\n".join(description) + '\n')
            file.write('Tempo Total: {0}\n'.format(t1 - t0))
            file.close()
          else:
            print('Skip ❌ ' + path_save)