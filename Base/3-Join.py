from glob import glob
from shutil import move

def sort(x):
    return int(x.split('/')[-1].split('.')[0])

path = "./"
classes = ["2Reais", "5Reais", "10Reais", "20Reais", "50Reais", "100Reais"]
for c in classes:
    for estate in glob(path + c + "/*"):
        for folder in glob(estate + "/*"):
            paths = sorted(glob(folder + "/*.jpg"), key = sort)
            tam = len(glob(estate + "/*.jpg")) + 1
            print(tam)
            for src in paths:
                print(src)
                des = estate + '/' + str(tam) + '.jpg'
                tam = tam + 1
                move(src,des)
                print(des)
        paths = sorted(glob(estate + "/*.jpg"), key = sort)
        tam = len(glob(c + "/*.jpg")) + 1
        print(tam)
        for src in paths:
            print(src)
            des = c + '/' + str(tam) + '.jpg'
            tam = tam + 1
            move(src,des)
            print(des)