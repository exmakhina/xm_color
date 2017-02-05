#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Guesstimat0rs for image dimensions

import sys, io

import numpy as np

def guess_2dbuffer_dimensions(buf, sx=1, sy=1):
	"""
	Guesstimate 2D buffer dimensions by finding factors that
	multiply to the buffer length, and finding the candidate buffer
	minimizing the total variation (eg. gradient).

	:param buf: input buffer (1D)
	:param sx: horizontal step for TV computation
	:param sy: vertical step for TV computation
	:return: reshaped buffer or None, and list of candidate values in increasing order
	"""

	candidates = []
	l = len(buf)
	for n in range(1, l):
		q, r = divmod(l, n)
		if r == 0 and (n % sx) == 0 and (q % sy) == 0 and q >= sy and n >= sx:
			candidates.append((n,q))

	res = []
	for w, h in candidates:
		def check_candidate():
			img = buf.reshape((h, w))
			tv = 0
			for ix in range(1, 1+sx):
				for iy in range(1, 1+sy):
					try:
						g = np.gradient(img[iy::sy,ix::sx])
					except ValueError:
						#print("Not possible: %s" % (str((h, w, iy,sy,ix,sx))))
						return
					tv += np.sum(g)
			res.append((tv, w, h))
		check_candidate()

	res.sort()

	if len(res) == 0:
		return None, res

	return buf.reshape((res[0][2], res[0][1])), res


def guess_image_dimensions(buf):
	"""
	Guesstimate image dimensions of a buffer.

	:param buf: input buffer (1D)
	:return: reshaped buffer or None, and list of candidate values in increasing order
	"""

	res = []
	for sx, sy in (
	 (1, 1),
	 (3, 1),
	 (6, 1),
	 (4, 1),
	 (2, 2),
	 (4, 4),
	 ):
		s, candidates = guess_2dbuffer_dimensions(buf, sx, sy)
		for tv, w, h in candidates:
			res.append((tv, sx, sy, w, h))

	res.sort()

	if len(res) == 0:
		return None, res

	tv, sx, sy, w, h = res[0]

	if sx == 3 and sy == 1:
		return buf.reshape((h, w//3, 3)), res
	elif sx == 4 and sy == 1:
		return buf.reshape((h, w//4, 4)), res
	else:
		return buf.reshape((h, w)), res


if __name__ == '__main__':

	import cv2

	img = cv2.imread("2016-06-03.jpg")#[...,0]
	buf = img.flatten()

	if 0:
		with open("pouet.raw", "rb") as f:
			data = f.read()

		buf = np.fromstring(data, dtype=np.uint8)

	img, candidates = guess_2dbuffer_dimensions(buf)#, sx=3)

	for tv, w, h in candidates:
		print("%f %dx%d" % (tv, w, h))

	cv2.imwrite("gen/tmp-guess-0-2dbuf.jpg", img)

	img, candidates = guess_image_dimensions(buf)
	#print(img.shape)

	for tv, sx, sy, w, h in candidates:
		print("%f %d/%d %dx%d" % (tv, sx, sy, w, h))

	cv2.imwrite("gen/tmp-guess-1-img.jpg", img)
