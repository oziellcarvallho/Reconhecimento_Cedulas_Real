from pandas import read_csv, unique


path = './Base/'

metadata = read_csv(path + 'Base_Metadata.csv')

print(metadata)

print('Número total de amostras da base: ', len(metadata))

print('\nClasses:', unique(metadata['Classe']))

print('\nQuantidade de Classes: ', len(unique(metadata['Classe'])))

print('\nDistribuição de Amostras por Classe:')
print(metadata['Classe'].value_counts())

print('\nDistribuição de Amostras por Set:')
print(metadata['Set'].value_counts())

print('\nTreino: ')
print(metadata[metadata['Set'] == 'Treino']['Classe'].value_counts())

print('\nTeste: ')
print(metadata[metadata['Set'] == 'Teste']['Classe'].value_counts())

print('\nProfundidade de Bits:')
print(metadata['Bit_Depth'].value_counts())

print('\nFormato:')
print(metadata['Format'].value_counts())

print('\nModo:')
print(metadata['Mode'].value_counts())

print('\nWidth:')
print('Distribuição:')
print(metadata['Width'].value_counts())
print('Menor: ', min(metadata['Width']))
print('Maior: ', max(metadata['Width']))

print('\nHeight:')
print('Distribuição:')
print(metadata['Height'].value_counts())
print('Menor: ', min(metadata['Height']))
print('Maior: ', max(metadata['Height']))