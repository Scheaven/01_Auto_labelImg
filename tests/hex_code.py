#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/10 15:18
# @Author  : Scheaven
# @File    :  hex_code.py
# @description:
'''
测试大端编码

'''


import binascii
import sys
print("本机存储模式是{} Endian.\n".format(sys.byteorder.capitalize()))

content = "待传输中文文本"  # content是str类型
print("原文本为：", content, '\n')

# 先看最终代码
result = binascii.b2a_hex(content.encode('utf-16-be'))
print("编码后文本为：", result, '\n')
content = binascii.a2b_hex(result).decode('utf-16-be')
print("解码后文本为：", content, '\n')

# ————————————————————————————————————————————————
# 下面开始解释代码
# 解释编码
coded = content.encode('utf-16-be') # coded是bytes类型，uft-16就不解释了，be是Big Endian的缩写。
print("使用utf-16-be编码后为：", coded, '\n')
result = binascii.b2a_hex(coded) # b2a应该是bytes to ascii，hex就是16进制的意思，b2a_hex方法和hexlify方法是相同的。
print("utf-16-be转为hex为：", result, '\n')

# 解释解码
coded = binascii.a2b_hex(result) # 同理a2b应该是ascii to bytes，hex也是16进制的意思，a2b_hex方法和unhexlify方法是相同的。
print("hex转为utf-16-be为：", coded, '\n')
content = coded.decode('utf-16-be')
print("utf-16-be解码后的原文本为：", content, '\n')
