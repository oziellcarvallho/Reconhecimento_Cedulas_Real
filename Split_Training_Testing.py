from sklearn.model_selection import train_test_split
from glob import glob
from numpy import save, load

path = './Base/'
classes = ['2', '5', '10', '20', '50', '100']
DTT = {}

for classe in classes:
  data = glob(path + classe +'Reais/*.jpg')
  print(classe)
  labels = [classe] * len(data)
  print('-------------------------')
  print('Caminhos: ', len(data))
  print('Classe: ', len(labels))
  data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1)
  print('Caminhos Treino: ', len(data_train))
  print('Caminhos Teste: ', len(data_test))
  print('Classe Treino: ', len(labels_train))
  print('Classe Teste: ', len(labels_test))
  print('-------------------------')
  p = {}
  p['Treino'] = data_train
  p['Teste'] = data_test
  DTT[classe] = p

print(len(DTT)) #6
print(len(DTT['2'])) #2
print(len(DTT['2']['Treino'])) #806

# Save
save(path + 'Data_Treino_Teste.npy', DTT) 

# Load
read_dictionary = load(path + 'Data_Treino_Teste.npy', allow_pickle=True).item()
print(len(read_dictionary)) #6
print(len(read_dictionary['2'])) #2
print(len(read_dictionary['2']['Treino'])) #806