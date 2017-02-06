#!/usr/bin/env python
# -*- coding: utf-8 vi:noet

"""

Generic color conversions, meant to be easy to understand and to operate
on provided data. OpenCV does have color conversion routines, but they may
be unusable in numerical optimization code.


"""


import numpy as np




"""
Conversion between RGB and YCbCr with BT.601
"""

def rgb_to_ycbcr_bt601_fs(rgb):
	R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
	yuv = np.zeros_like(rgb)
	Y = 0.114*B + 0.587*G + 0.299*R
	U = 0.5*B - 0.331264108352144*G - 0.168735891647856*R
	V = -0.0813124108416548*B - 0.418687589158345*G + 0.5*R
	yuv[...,0] = Y
	yuv[...,1] = U
	yuv[...,2] = V
	return yuv

def ycbcr_bt601_fs_to_rgb(yuv):
	Y, U, V = yuv[...,0], yuv[...,1], yuv[...,2]
	rgb = np.zeros_like(yuv)
	R = 1.402*V + Y
	G = -0.344136286201022*U - 0.714136286201022*V + Y
	B = 1.772*U + Y
	rgb[...,0] = R
	rgb[...,1] = G
	rgb[...,2] = B
	return rgb



"""
Conversion between RGB and YCbCr with BT.709
"""

def rgb_to_ycbcr_bt709_fs(rgb):
	R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
	yuv = np.zeros_like(rgb)
	Y = 0.0722*B + 0.7152*G + 0.2126*R
	U = 0.5*B - 0.38542789394266*G - 0.11457210605734*R
	V = -0.0458470916941834*B - 0.454152908305817*G + 0.5*R
	yuv[...,0] = Y
	yuv[...,1] = U
	yuv[...,2] = V
	return yuv

def ycbcr_bt601_fs_to_rgb(yuv):
	Y, U, V = yuv[...,0], yuv[...,1], yuv[...,2]
	rgb = np.zeros_like(yuv)
	R = 1.5748*V + Y
	G = -0.187324272930649*U - 0.468124272930649*V + Y
	B = 1.8556*U + Y
	rgb[...,0] = R
	rgb[...,1] = G
	rgb[...,2] = B
	return rgb




def r709_gamma(rgb):
	"""
	Convert from linear RGB to R.709 gamma RGB
	:param rgb: array with values in [0,1]
	"""
	srgb = np.zeros_like(rgb)
	t0 = 0.0018
	alpha = 0.099
	srgb[rgb<=t0] = 4.5 * rgb[rgb<=t0]
	srgb[rgb>t0] = (1+alpha)*(rgb[rgb>t0]**(0.45)) - alpha
	return srgb


def r709_invgamma(srgb):
	"""
	Convert from R.709 gamma RGB to linear RGB
	:param srgb: array with values in [0,1]
	"""
	rgb = np.zeros_like(srgb)
	t0 = 0.081
	alpha = 0.055
	rgb[srgb<=t0] = 1/4.5 * srgb[srgb<=t0]
	rgb[srgb>t0] = ((srgb[srgb>t0]+alpha)/(1+alpha))**(1/0.45)
	return rgb



"""
Conversion between RGB and YUV with BT.601

Coefficients computed from conversions_computations.py
"""

def rgb_to_yuv_bt601_fs(rgb):
	R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
	yuv = np.zeros_like(rgb)
	Y = 0.114*B + 0.587*G + 0.299*R
	U = 0.436*B - 0.28886230248307*G - 0.14713769751693*R
	V = -0.100014265335235*B - 0.514985734664765*G + 0.615*R
	yuv[...,0] = Y
	yuv[...,1] = U
	yuv[...,2] = V
	return yuv

def yuv_bt601_fs_to_rgb(yuv):
	Y, U, V = yuv[...,0], yuv[...,1], yuv[...,2]
	rgb = np.zeros_like(yuv)
	R = 1.13983739837398*V + Y
	G = -0.39465170435897*U - 0.580598606667498*V + Y
	B = 2.03211009174312*U + Y
	rgb[...,0] = R
	rgb[...,1] = G
	rgb[...,2] = B
	return rgb



def ccm_3x4(rgb, xr, xg, xb, ox, yr, yg, yb, oy, zr, zg, zb, oz):
	"""
	"""
	xyz = np.zeros_like(rgb)
	xyz[:,:,0] = xr * rgb[:,:,0] + \
				 xg * rgb[:,:,1] + \
				 xb * rgb[:,:,2] + \
				 ox
	xyz[:,:,1] = yr * rgb[:,:,0] + \
				 yg * rgb[:,:,1] + \
				 yb * rgb[:,:,2] + \
				 oy
	xyz[:,:,2] = zr * rgb[:,:,0] + \
				 zg * rgb[:,:,1] + \
				 zb * rgb[:,:,2] + \
				 oz
	return xyz



"""
sRGB stuff.

https://en.wikipedia.org/wiki/SRGB
sRGB is an RGB color space that uses the ITU-R BT.709 primaries,

"""

def srgb_gamma(rgb):
	"""
	Convert from linear RGB to sRGB
	"""
	srgb = np.zeros_like(rgb)
	t0 = 0.0031308
	alpha = 0.055
	srgb[rgb<=t0] = 12.92 * rgb[rgb<=t0]
	srgb[rgb>t0] = (1+alpha)*(rgb[rgb>t0]**(1/2.4)) - alpha
	return srgb


def srgb_invgamma(srgb):
	"""
	Convert from sRGB to linear RGB
	"""
	rgb = np.zeros_like(srgb)
	t0 = 0.04045
	alpha = 0.055
	rgb[srgb<=t0] = 1/12.92 * srgb[srgb<=t0]
	rgb[srgb>t0] = ((srgb[srgb>t0]+alpha)/(1+alpha))**2.4
	return rgb





"""
CIE LAB stuff
"""

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
	x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]

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

	lab[...,0] = l
	lab[...,1] = a
	lab[...,2] = b

def cv2_rgbswap(img):
	"""
	OpenCV uses RGB24 ordered in BGR, not RGB.
	"""
	res = np.empty_like(img)
	res[...,0] = img[...,2]
	res[...,1] = img[...,1]
	res[...,2] = img[...,0]
	return res

