"""
    先来说一下jpg图片和png图片的区别
    jpg格式:是有损图片压缩类型,可用最少的磁盘空间得到较好的图像质量
    png格式:不是压缩性,能保存透明等图

"""
from PIL import Image
import cv2 as cv
import os,io
import PIL.Image
import tensorflow as tf

def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w), int(h)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=100)
            #os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=100)
            #os.remove(PngPath)
        print(outfile)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


if __name__ == '__main__':
    li=os.listdir("G:/qqPoker/")
    for i in li:
        #print(i)
        with tf.gfile.GFile("G:/qqPoker/"+i, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        #print(image.format)
        if image.format != 'JPEG':
            print(image.format)
            print(i)
            PNG_JPG("G:/qqPoker/"+i)

