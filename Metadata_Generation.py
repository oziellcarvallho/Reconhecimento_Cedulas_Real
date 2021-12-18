from numpy import load, savetxt
from PIL import Image
from os import chdir

path = './Base/'

#L (Pixels de 8 bits, preto e branco)
# Name, Set, Format, Mode, Bit_Depth, Width, Height, Classe, Id_Classe

dtt = load(path + 'Data_Treino_Teste.npy', allow_pickle=True).item()

def image_features(file_path):
    print(file_path)
    features = []
    image = Image.open(file_path)
    features.append(image.filename.split('/')[-1])
    features.append(image.format)
    features.append(image.mode)
    features.append(image.bits)
    features.append(image.width)
    features.append(image.height)
    image.close()
    return features

def run():
    classes = ['2', '5', '10', '20', '50', '100']
    folders = ['Treino', 'Teste']
    cedulas_metadata = []
    for classe in classes:
        for folder in folders:
            for file_path in dtt[classe][folder]:
                features = image_features(file_path)
                cedulas_metadata.append([features[0], folder, features[1], features[2], features[3], features[4], features[5], classe + ' Reais', int(classe)])

    return cedulas_metadata

metadata = run()

#Salvando os metadados em um .csv
chdir(path)
header = 'Name,Set,Format,Mode,Bit_Depth,Width,Height,Classe,Id_Classe'
savetxt("Base_Metadata.csv", metadata, delimiter=",", fmt='%s', header=header)