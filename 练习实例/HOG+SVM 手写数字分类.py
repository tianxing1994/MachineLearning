"""
参考链接:
https://blog.csdn.net/wsp_1138886114/article/details/82948880
http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html

数据集下载地址:
http://yann.lecun.com/exdb/mnist/
链接: https://pan.baidu.com/s/1aN-LSzL7y2RF2RDkjddcjA 密码: iays
"""
import os
import struct

import cv2 as cv
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def load_mnist(path):
    labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    images_path = os.path.join(path, 'train-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def image2hog(image):
    # fd = hog(image,
    #          orientations=9,
    #          pixels_per_cell=(14, 14),
    #          cells_per_block=(1, 1),
    #          visualize=False)
    fd = hog(image,
             orientations=9,
             pixels_per_cell=(7, 7),
             cells_per_block=(2, 2),
             visualize=False)
    return fd


def load_in_hog_feature(path):
    ret = list()
    images, labels = load_mnist(path)
    for feature in images:
        image = np.reshape(feature, newshape=(28, 28))
        fd = image2hog(image)
        ret.append(fd)
    return np.array(ret), labels


def demo1():
    """训练 SVC 分类模型."""
    path = "../dataset/handwritten-digit"
    hog_features, labels = load_in_hog_feature(path)
    clf = LinearSVC()
    clf.fit(hog_features, labels)
    joblib.dump(clf, "../dataset/handwritten-digit/temp/digits_cls.pkl", compress=3)
    return


def demo2():
    """加载图片, 测试分类效果."""
    clf = joblib.load("../dataset/handwritten-digit/temp/digits_cls.pkl")
    image_dir = "../dataset/handwritten-digit/t10k-images-bmp"
    image_name_list = os.listdir(image_dir)
    np.random.shuffle(image_name_list)
    for image_name in image_name_list:
        image_path = os.path.join(image_dir, image_name)
        image = cv.imread(image_path)
        # image_hog_fd = hog(image, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        image_hog_fd = hog(image, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualize=False)
        nbr = clf.predict(np.array([image_hog_fd], 'float64'))
        print(nbr)
        show_image(image)
    return


if __name__ == '__main__':
    # demo1()
    demo2()
