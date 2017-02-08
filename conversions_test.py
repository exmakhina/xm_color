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


def test_ycbcr():
	img = chart_rgb24.get_RGB24_test_img()

	img = np.float64(img)/255
	imwrite("yuv-601-00-rgb24.png", img)

	img_yuv = conversions.rgb_to_ycbcr_bt601_fs(img)
	img_rgb = conversions.ycbcr_bt601_fs_to_rgb(img_yuv)
	imwrite("yuv-601-02-rgb.png", img_rgb)

	d = np.abs(img-img_rgb)
	_min, _max, _mean = np.min(d), np.max(d), np.mean(d)
	print("Delta min=%f max=%f mean=%f" % (_min, _max, _mean))


	img = chart_rgb24.get_RGB24_test_img()

	img = np.float64(img)/255
	imwrite("yuv-706-00-rgb24.png", img)

	img_yuv = conversions.rgb_to_ycbcr_bt709_fs(img)
	img_rgb = conversions.ycbcr_bt709_fs_to_rgb(img_yuv)
	imwrite("yuv-709-02-rgb.png", img_rgb)

	d = np.abs(img-img_rgb)
	_min, _max, _mean = np.min(d), np.max(d), np.mean(d)
	print("Delta min=%f max=%f mean=%f" % (_min, _max, _mean))



if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser(
	 description="Conversion tests",
	)

	subparsers = parser.add_subparsers(
	 help='the command; type "%s COMMAND -h" for command-specific help' % sys.argv[0],
	 dest='command',
	)

	for k, v in locals().items():
		if k.startswith("test_"):
			subp = subparsers.add_parser(
			 k,
			 help="",
			)

	try:
		import argcomplete
		argcomplete.autocomplete(parser)
	except:
		pass

	args = parser.parse_args()

	if 0:
		pass

	for k, v in locals().items():
		if args.command == k:
			v()
