#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：relation 
@File    ：str_util.py
@Author  ：hbx
@Date    ：2022/2/2 18:03 
'''

def format_str(text: str, lower = False, half = True) -> str:
    if lower:
        text = text.lower()
    if half:
        text = full2half(text)
    return text


#全角转成半角
def full2half(s):
    n = ''
    for char in s:
        num = ord(char)
        if num == 0x3000:        #将全角空格转成半角空格
            num = 32
        elif 0xFF01 <=num <= 0xFF5E:       #将其余全角字符转成半角字符
            num -= 0xFEE0
        num = chr(num)
        n += num
    return n



def half2full(s):
    n = ''
    for char in s:
        num = ord(char)
        if(num == 32):             #半角空格转成全角
            num = 0x3000
        elif 33 <= num <= 126:
            num += 65248           #16进制为0xFEE0
        num = chr(num)
        n += num
    return n