import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    list=[]
    tree = ET.parse(path)
    root = tree.getroot()
    for member in root.findall('object'):
        value = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 member[0].text,
                 int(member[4][0].text),
                 int(member[4][1].text),
                 int(member[4][2].text),
                 int(member[4][3].text)
                 )
        list.append(value)
    return list


def main():
    for folder in ['train.txt','test.txt','val.txt']:
        xml_list = []
        txt_path='G:/more/'+folder
        with open(txt_path, "r") as f:
            for line in f:
                tmp = line.rstrip()
                list = xml_to_csv('G:/more/xml/'+tmp+'.xml')
                for item in list:
                    xml_list.append(item)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv(('G:/PycharmProjects/models-master/dataset/faster_rcnn_inception_v2_coco/' + folder + '_labels.csv'), index=None)

        print('Successfully converted xml to csv.')


main()
