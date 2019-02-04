#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Guesstimat0rs for image dimensions

import sys, io

import numpy as np
def total_variation(img):
	"""
	"""

	if 1:
		gy = np.diff(img, axis=0)
		tv = np.mean((gy**2)**0.5)
		return tv

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

	max_ar = 10.0

	candidates = []
	l = len(buf)
	for n in range(1, l):
		q, r = divmod(l, n)
		if r == 0 and (n % sx) == 0 and (q % sy) == 0 and q >= sy and n >= sx:

			if n > max_ar * q or q > max_ar * n:
				continue

			candidates.append((n,q))

	res = []
	for w, h in candidates:
		def check_candidate():
			img = np.float64(buf.reshape((h, w)))
			tv = 0
			for ix in range(1, 1+sx):
				for iy in range(1, 1+sy):

					reint = img[iy::sy,ix::sx]

					tv += total_variation(img)

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

	import argparse
	import cv2


	parser = argparse.ArgumentParser(
	 description="image dimension guesser",
	)

	subparsers = parser.add_subparsers(
	 help='the command; type "%s COMMAND -h" for command-specific help' % sys.argv[0],
	 dest='command',
	)

	parser_log = subparsers.add_parser(
	 "raw",
	 help="guess raw buffer dimensions",
	)

	parser_log.add_argument("path",
	 help="file to process",
	)

	parser_log.add_argument("--dtype",
	 type=np.dtype,
	 default=np.uint8,
	)

	parser_log.add_argument("--out",
	)

	parser_log = subparsers.add_parser(
	 "image",
	 help="guess raw buffer dimensions",
	)

	parser_log.add_argument("path",
	 help="file to process",
	)

	parser_log.add_argument("--out",
	)

	try:
		import argcomplete
		argcomplete.autocomplete(parser)
	except:
		pass

	args = parser.parse_args()

	if args.command == "raw":

		with open(args.path, "rb") as f:
			data = f.read()

		buf = np.fromstring(data, dtype=args.dtype)

		img, candidates = guess_2dbuffer_dimensions(buf)


		if args.out:
			import cv2
			cv2.imwrite(args.out, img)

	if args.command == "image":
		import cv2

		buf = cv2.imread(args.path)
		if len(buf.shape) > 2:
			buf = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
		buf = buf.reshape((-1,))

		img, candidates = guess_2dbuffer_dimensions(buf)

		for tv, w, h in candidates:
			print("%f %dx%d" % (tv, w, h))

		if args.out:
			img = np.uint8(cv2.normalize(np.float32(img), dst=None, norm_type=cv2.NORM_MINMAX, alpha=0, beta=255))
			cv2.imwrite(args.out, img)

	if 0:
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
