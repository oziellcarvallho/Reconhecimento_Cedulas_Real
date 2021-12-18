from cv2 import imread, xfeatures2d, MSER_create
from pickle import dump
from os import mkdir, path
import numpy as np
import time
from statistics import median

n_test = '4'
path_dataset = './Base/'
path_pickles = './Tests/Test' + n_test +'/pickles/MSER/'
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

classes = ['2', '5', '10', '20', '50', '100']
stages = ['Treino', 'Teste']
calculos = ['mean', 'median']
for calc in calculos:
  description = []
  mser = MSER_create(_delta = 2)
  description.append('Nome Padrão:{0}'.format(mser.getDefaultName()))
  description.append('Delta:{0}'.format(mser.getDelta()))
  description.append('MaxArea:{0}'.format(mser.getMaxArea()))
  description.append('MinArea:{0}'.format(mser.getMinArea()))
  description.append('Pass2Only:{0}'.format(mser.getPass2Only()))
  description.append('Descriptor Size:{0}-'.format(mser.descriptorSize()))
  surf = xfeatures2d.SURF_create(hessianThreshold = 200, nOctaves = 22, nOctaveLayers = 21)
  description.append('Descriptor_Size:{0}'.format(surf.descriptorSize()))
  description.append('HessianThreshold:{0}'.format(surf.getHessianThreshold()))
  description.append('NOctaveLayers:{0}'.format(surf.getNOctaveLayers()))
  description.append('NOctaves:{0}'.format(surf.getNOctaves()))
  description.append(calc)
  print("\n".join(description))        
  path_save = path_pickles + 'Surf_MSER' + "_".join(description)      
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
          keys = mser.detect(img, None)
          kp, des = surf.compute(img, keys)
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
    file.write('\nSurf_MSER\n' +"\n".join(description) + '\n')
    file.write('Tempo Total: {0}\n'.format(t1 - t0))
    file.close()
  else:
    print('Skip ❌ ' + path_save)