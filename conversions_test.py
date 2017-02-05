#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Testing for conversion routines

import sys, os

import numpy as np
import cv2

from xm_color import conversions
from xm_color import chart_rgb24

def imwrite(name, img):
	wd = "gen/tmp-conversions-test"
	if not os.path.exists(wd):
		os.makedirs(wd)

	if img.dtype in (np.float32, np.float64):
		img = np.uint8(np.round(img*255))

	cv2.imwrite(os.path.join(wd, name), img)

def test_gamma():
	"""
	Check some gamma stuff.
	"""

	img = chart_rgb24.get_RGB24_test_img()

	img = np.float64(img)/255
	imwrite("00-rgb24.png", img)
	img = conversions.srgb_gamma(img)
	imwrite("01-rgb24-gamma.png", img)
	img = conversions.srgb_invgamma(img)
	imwrite("02-rgb24-gamma_invgamma.png", img)
	img = conversions.srgb_invgamma(img)
	imwrite("03-rgb24-invgamma.png", img)

if __name__ == "__main__":
	test_gamma()

