#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Charts that use all RGB24 colors

import numpy as np

def get_RGB24_test_img():
	"""
	:return: simplest image that contains all (RGB24) possible colors
	"""
	a = np.arange((1<<24), dtype="<I")
	data = a.tobytes()
	a = np.fromstring(data, dtype=np.uint8).reshape((4096, 4096, 4))
	data = a[...,0:3].tobytes()
	a = np.fromstring(data, dtype=np.uint8)
	data = a.reshape((4096, 4096, 3))
	return data

