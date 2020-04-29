import cv2
import numpy as np
import os
from functools import reduce
from PIL import Image


def isLike(file1, file2):
    image1 = cv2.imread(os.path.join(os.getcwd(), 'tmp/upload/' + file1))
    image2 = cv2.imread(os.path.join(os.getcwd(), 'tmp/upload/' + file2))
    difference = cv2.subtract(image1, image2)
    result = not np.any(difference)  # if difference is all zeros it will return False

    if result is True:
        return 1
        print("两张图片一样")
    else:
        return 0
        print("两张图片不一样")


def isExsit(file):
    return os.path.exists(os.path.join(os.getcwd(), 'tmp/upload/' + file))


# 这种算法的优点是简单快速，不受图片大小缩放的影响，
# 缺点是图片的内容不能变更。如果在图片上加几个文字，它就认不出来了。
# 所以，它的最佳用途是根据缩略图，找出原图。


# 计算图片的局部哈希值--pHash
def phash(img):
    """
    :param img: 图片
    :return: 返回图片的局部hash值
    """
    img = img.resize((8, 8), Image.ANTIALIAS).convert('L')
    avg = reduce(lambda x, y: x + y, img.getdata()) / 64.
    hash_value = reduce(lambda x, y: x | (y[1] << y[0]), enumerate(map(lambda i: 0 if i < avg else 1, img.getdata())),
                        0)

    return hash_value


# 自定义计算两个图片相似度函数局部敏感哈希算法
def phash_img_similarity(img1_path, img2_path):
    img1_path = os.path.join(os.getcwd(), 'tmp/upload/' + img1_path)
    img2_path = os.path.join(os.getcwd(), 'tmp/upload/' + img2_path)
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """

    # 读取图片
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # 计算两个图片的局部哈希值

    # 计算局部敏感哈希值
    img1_phash = str(phash(img1))
    img2_phash = str(phash(img2))

    # 打印局部敏感哈希值
    print(img1_phash)
    print(img2_phash)

    # 计算汉明距离

    distance = bin(phash(img1) ^ phash(img2)).count('1')
    print(distance)

    print(max(len(bin(phash(img1))), len(bin(phash(img1)))))

    similary = 1 - distance / max(len(bin(phash(img1))), len(bin(phash(img1))))
    print("两张图片相似度为:%s" % similary)
    return similary


if __name__ == '__main__':
    img1_path = r"G:\PycharmProjects\flaskTest\tmp\upload\a253bbee-aced-11e9-b0ef-005056c00004.jpg"
    img2_path = r"G:\PycharmProjects\flaskTest\tmp\upload\95ca3350-aced-11e9-b68d-005056c00004.jpg"
    similary = phash_img_similarity(img1_path, img2_path)
