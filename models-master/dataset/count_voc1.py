# coding:utf-8
import xml.etree.cElementTree as ET
import os
from collections import Counter
import shutil


# Counter({'towCounter({'tower': 3074, 'windpower': 2014, 'thermalpower': 689, 'hydropower': 261, 'transformer': 225})
# total_num: 6263


def count(pathdir, despath):
    item={}
    tmp=""
    path = pathdir
    for index, xml in enumerate(os.listdir(path)):
        # print(str(index) + ' xml: '+ xml)
        root = ET.parse(os.path.join(path, xml))
        objects = root.findall('object')


        for ob in objects:
            key=ob.find('name').text

            if tmp!=key:

                if key in item:
                    item.update({key:int(item.get(key))+1})
                    tmp=key
                else:
                    item.update({key:1})
    print(item)



if __name__ == '__main__':
    # pathdirs = list(set(os.listdir('./')) ^ set(['tools','count.py']))
    # print(pathdirs)
    # for pathdir in pathdirs:
    pathdir = 'G:/PycharmProjects/models-master/dataset/test_images'
    despath = '/transformer/'
    count(pathdir, despath)
