#!/usr/bin/env python
# -*- coding: utf-8 vi:noet

"""

Generic color conversions, meant to be easy to understand and to operate
on provided data. OpenCV does have color conversion routines, but they may
be unusable in numerical optimization code.


"""


import numpy as np


"""
Coefficients computed from conversions_computations.py
"""

def rgb_to_ycbcr_bt601_fs(rgb):
	R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
	yuv = np.zeros_like(rgb)
	Y = 0.114*B + 0.587*G + 0.299*R
	U = 0.436*B - 0.28886230248307*G - 0.14713769751693*R
	V = -0.100014265335235*B - 0.514985734664765*G + 0.615*R
	yuv[...,0] = Y
	yuv[...,1] = U
	yuv[...,2] = V
	return yuv

def ycbcr_bt601_fs_to_rgb(yuv):
	Y, U, V = yuv[...,0], yuv[...,1], yuv[...,2]
	rgb = np.zeros_like(yuv)
	R = 1.13983739837398*V + Y
	G = -0.39465170435897*U - 0.580598606667498*V + Y
	B = 2.03211009174312*U + Y
	rgb[...,0] = R
	rgb[...,1] = G
	rgb[...,2] = B
	return rgb

