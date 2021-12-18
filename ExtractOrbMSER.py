from cv2 import imread, ORB_create, MSER_create
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

def vetor_create(orb, x):
  a = []
  if (x == 'mean'):
    for j in range(orb.descriptorSize()):
      a.append(des.transpose()[j].mean())
  if (x == 'median'):
    for j in range(orb.descriptorSize()):
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
  orb = ORB_create(edgeThreshold = 20, patchSize = 20)
  description.append('Descriptor_Size:{0}'.format(orb.descriptorSize()))
  description.append('EdgeThreshold:{0}'.format(orb.getEdgeThreshold()))
  description.append('FastThreshold:{0}'.format(orb.getFastThreshold()))
  description.append('FirstLevel:{0}'.format(orb.getFirstLevel()))
  description.append('MaxFeatures:{0}'.format(orb.getMaxFeatures()))
  description.append('NLevels:{0}'.format(orb.getNLevels()))
  description.append('PatchSize:{0}'.format(orb.getPatchSize()))
  description.append(calc)
  print("\n".join(description))        
  path_save = path_pickles + 'ORB_MSER' + "_".join(description)      
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
          kp, des = orb.compute(img, keys)
          print(len(kp))
          print(len(des))
          a = vetor_create(orb, calc)
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
    file.write('\nORB_MSER\n' +"\n".join(description) + '\n')
    file.write('Tempo Total: {0}\n'.format(t1 - t0))
    file.close()
  else:
    print('Skip ❌ ' + path_save)       
          
          