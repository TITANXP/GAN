import cv2
import os.path
import sys
from glob import glob
import random
import datetime

def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    try:
        if not os.path.isfile(cascade_file):
            raise RuntimeError("%s: not found" % cascade_file)
        cascade = cv2.CascadeClassifier(cascade_file)
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(48, 48))
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y: y + h, x:x + w, :]
            face = cv2.resize(face, (96, 96))
            # save_filename = '%s-%d.jpg' % (os.path.basename(filename).replace('yande.re ', '').split('.')[0], i)
            save_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(random.randint(1, 999)) + '.jpg'
            print(save_filename)
            cv2.imwrite("faces/" + save_filename, face)
    except:
        pass


if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob(r'F:\Start_Here_Mac.app\Contents\yande\*.png')
    # file_list = glob(r'image\*')

    for filename in file_list:
        detect(filename)


