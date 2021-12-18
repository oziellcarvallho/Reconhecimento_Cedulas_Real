from glob import glob
from random import randrange
from numpy import ones, array
from skimage.util import random_noise
from cv2 import imread, imwrite, blur, add, subtract

#Função responsável por inserir ruídos randomincos do tipo sal e pimenta na imagem. (Entrada: Imagem, Saída: Imagem com ruidos do tipo sal e pimenta)
def ruidosImg(image):
    noise_img = random_noise(image, mode='s&p', amount=0.04)
    return array(255*noise_img, dtype = 'uint8')

#Função responsável por incrementar brilho a imagem. (Entrada: Imagem, Saída: Imagem com brilho)
def brightnessUp(image):
    bright = ones(image.shape, dtype="uint8") * 50
    bright_up = add(image, bright)
    return bright_up

#Função responsável por decrementar brilho a imagem. (Entrada: Imagem, Saída: Imagem com brilho)
def brightnessDown(image):
    bright = ones(image.shape, dtype="uint8") * 50
    bright_down = subtract(image, bright)
    return bright_down

#Função responsável por aplicar um filtro mediana na imagem. (Entrada: Imagem, Saída: Imagem com filtro mediana)
def blurImg(image):
    #randrange(3,5,2)
    k_size = (3, 3)
    img_blur = blur(image, k_size)
    return img_blur

def sort(x):
    return int(x.split('/')[-1].split('.')[0])

path = "./"
classes = ["2Reais", "5Reais", "10Reais", "20Reais", "50Reais", "100Reais"]
for c in classes:
    for estate in glob(path + c + "/*"):
        for folder in glob(estate + "/*"):
            paths = sorted(glob(folder + "/*.jpg"), key = sort)
            tam = len(paths) + 1
            print(tam)
            for src in paths:
                print(src)
                a = ''.join(src.split('/')[-2:-1])
                des = estate + '/' + '/'.join(src.split('/')[-2:-1]) + '/' + str(tam) + '.jpg'
                tam = tam + 1
                print('ruidos_img')
                imwrite(des, ruidosImg(imread(src)))
                print(des)

                print(src)
                a = ''.join(src.split('/')[-2:-1])
                des = estate + '/' + '/'.join(src.split('/')[-2:-1]) + '/' + str(tam) + '.jpg'
                tam = tam + 1
                print('brightness up')
                imwrite(des, brightnessUp(imread(src)))
                print(des)

                print(src)
                a = ''.join(src.split('/')[-2:-1])
                des = estate + '/' + '/'.join(src.split('/')[-2:-1]) + '/' + str(tam) + '.jpg'
                tam = tam + 1
                print('brightness down')
                imwrite(des, brightnessDown(imread(src)))
                print(des)

                print(src)
                a = ''.join(src.split('/')[-2:-1])
                des = estate + '/' + '/'.join(src.split('/')[-2:-1]) + '/' + str(tam) + '.jpg'
                tam = tam + 1
                print('blur_img')
                imwrite(des, blurImg(imread(src)))
                print(des)

                print(src)
                a = ''.join(src.split('/')[-2:-1])
                des = estate + '/' + '/'.join(src.split('/')[-2:-1]) + '/' + str(tam) + '.jpg'
                tam = tam + 1
                print('ruidos_img + blur_img')
                imwrite(des, ruidosImg(blurImg(imread(src))))
                print(des)