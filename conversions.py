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



def xyz2lab(xyz, xn=100.0/95.047, yn=100.0/100.0, zn=100.0/108.883):
	"""
	Convert an image in XYZ color space to L*a*b* color space.

	:param xyz: image in xyz color space, must be a floating-point image.
	:param xn: X white point adjustment factor
	:param yn: Y white point adjustment factor
	:param zn: Z white point adjustment factor

	References:

	- https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions

	"""
	assert xyz.dtype in (np.float32, np.float64), xyz.dtype
	lab = np.zeros_like(xyz)
	x, y, z = cv2.split(xyz)

	delta = 6.0/29.0
	delta2 = (6.0**2)/(29**2)
	delta3 = (6.0**3)/(29**3)

	xb, xs = x > delta3, x <= delta3
	yb, ys = y > delta3, y <= delta3
	zb, zs = z > delta3, z <= delta3

	fx = np.zeros_like(x)
	fy = np.zeros_like(y)
	fz = np.zeros_like(z)
	fx[xb] = (x[xb]*xn)**(1./3)
	fx[xs] = (x[xs]*xn) / (3 * delta2) + 16./116
	fy[yb] = (y[yb]*yn)**(1./3)
	fy[ys] = (y[ys]*yn) / (3 * delta2) + 16./116
	fz[zb] = (z[zb]*zn)**(1./3)
	fz[zs] = (z[zs]*zn) / (3 * delta2) + 16./116

	l = 116 * fy - 16
	a = 500 * (fx - fy)
	b = 200 * (fy - fz)

	return cv2.merge((l, a, b))
