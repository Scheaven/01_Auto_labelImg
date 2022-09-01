#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   xml_tools.py
@Time    :   2021/11/02 18:36:31
@Author  :   Scheaven 
@Version :   1.0
@Contact :   snow_mail@foxmail.com
@License :   (C)Copyright 2021-2022, Scheaven 360
@Desc    :   None
'''

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

import xml.dom.minidom
from xml.etree import ElementTree as ET

class XMLTool(object):
    def __init__(self, xml_path, image_name, w, h) -> None:
        self.xml_path = xml_path

        node_root = Element('annotation')
        
        node_folder = SubElement(node_root, 'folder')
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = image_name + '.jpg'

        node_source = SubElement(node_root, "source")

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(w)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(h)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        self.tree=ET.ElementTree(node_root)
        self.tree.write(xml_path)

    def add_object(self, box, label):
        # root = self.dom.documentElement
        root = self.tree.getroot()
        node_object = SubElement(root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label
        node_difficult = SubElement(node_object, 'occluded')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))

        self.pretty_xml(root, '\t', '\n')
        self.tree.write(self.xml_path)
        # tree.write('sampleFileNew.xml')

    def pretty_xml(self, element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素    
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个    
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


    # tree = ElementTree.parse('movies.xml')  # 解析movies.xml这个文件
    # root = tree.getroot()  # 得到根元素，Element类
    # pretty_xml(root, '\t', '\n')  # 执行美化方法
    # tree.write('output.xml')


class UpdateXMLTool(object):
    def __init__(self, xml_path) -> None:
        self.xml_path = xml_path

        self.tree=ET.parse(xml_path)

    def add_object(self, box, label):
        root = self.tree.getroot()

        node_object =  ET.Element('object')
        node_name =  ET.SubElement(node_object, 'name')
        node_name.text = label
        node_difficult =  ET.SubElement(node_object, 'occluded')
        node_difficult.text = '0'

        node_bndbox =  ET.SubElement(node_object, 'bndbox')
        node_xmin =  ET.SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box[0]))
        node_ymin =  ET.SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box[1]))
        node_xmax =  ET.SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box[2]))
        node_ymax =  ET.SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box[3]))

        self.pretty_xml(node_object, '\t', '\n')
        root.append(node_object)
        self.tree.write(self.xml_path)

    def pretty_xml(self, element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
        if element:  # 判断element是否有子元素    
            if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
                element.text = newline + indent * (level + 1)
            else:
                element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
                # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
                # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
        temp = list(element)  # 将element转成list
        for subelement in temp:
            if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
                subelement.tail = newline + indent * (level + 1)
            else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个    
                subelement.tail = newline + indent * level
            self.pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作