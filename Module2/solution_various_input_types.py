#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:30:34 2017

@author: szabolcs
"""
from scipy import misc

file_path = "/Users/szabolcs/dev/git/DAT210x/Module2/"
file_name = "image.jpg"

img = misc.imread(file_path + file_name)

print(type(img))
print(img.shape, img.dtype)

img = img[::2, ::2]
print(img.shape)

img = (img / 255.0).reshape(-1, 3)
print(img.shape)

red = img[:,0]
green = img[:,1]
blue = img[:,2]

gray = (0.299 * red + 0.587 * green + 0.114 * blue)

print("img:", img.shape)
print("gray:", gray.shape)
