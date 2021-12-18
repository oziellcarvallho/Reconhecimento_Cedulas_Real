from os import path
from cv2 import imread, rotate, imwrite, ROTATE_180
from numpy import array
from glob import glob

ori = [1, 3, 5, 7, 9, 10, 13, 14, 17, 18, 21, 22]
des = [2, 4, 6, 8, 12, 11, 16, 15, 20, 19, 24, 23]
path = "./"

classes = ["2Reais", "5Reais", "10Reais", "20Reais", "50Reais", "100Reais"]
for c in classes:
    for estate in glob(path + c + "/*"):
        for folder in glob(estate + "/*"):
            for i in range(0, len(ori)):
                print("(", ori[i], " -> ", des[i], ")")
                path_ori = folder +"/"+ str(ori[i]) + ".jpg"
                path_des = folder +"/"+ str(des[i]) + ".jpg"
                img = imread(path_ori)
                rot_img = rotate(img, ROTATE_180)
                img_array = array(rot_img)
                print(img_array.shape)
                imwrite(path_des, rot_img)
                print(path_ori +" - "+ path_des)