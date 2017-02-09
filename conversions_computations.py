#!/usr/bin/env python
# -*- coding: utf-8 vi:noet

import numpy as np
import sympy
import cv2

from xm_color import conversions

def do_yuv():
	"""
	Color analog TV: perform YUV <-> RGB computations.

	The luma and chroma components in YUV are calculated from gamma corrected RGB.

	References:

	- https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601

	"""

	"""
	BT.601 defines the following constants:

	- W_R = 0.299
	- W_B = 0.114
	- W_G = 1 - W_R - W_B
	- U_{max} = 0.436
	- V_{max} = 0.615
	- Y = W_R * R + W_G * G + W_B * B
	- U = U_{max} * (B - Y) / (1 - W_B)
	- V = V_{max} * (R - Y) / (1 - W_R)

	"""

	R, G, B, W_R, W_G, W_B, U_max, V_max, Y, U, V \
	 = sympy.symbols("R, G, B, W_R, W_G, W_B, U_max, V_max, Y, U, V")

	e1 = sympy.Eq(W_R, 0.299)
	e2 = sympy.Eq(W_B, 0.114)
	e3 = sympy.Eq(W_G, 1 - W_R - W_B)
	e4 = sympy.Eq(U_max, 0.436)
	e5 = sympy.Eq(V_max, 0.615)
	e6 = sympy.Eq(U, U_max * (B - Y) / (1 - W_B))
	e7 = sympy.Eq(V, V_max * (R - Y) / (1 - W_R))
	e8 = sympy.Eq(Y, R * W_R + G * W_G + B * W_B)

	x = sympy.solve([e1, e2, e3], [W_R, W_B, W_G])
	print(x)
	for name in ("W_R", "W_G", "W_B"):
		for k, v in x.items():
			if k.name == name:
				print("%s = %s" % (k, v))

	"""
	RGB2YUV
	"""

	x = sympy.solve(
	 [e1, e2, e3, e4, e5, e6, e7, e8],
	 exclude=[R,G,B])
	assert len(x) == 1
	print(x)
	s = x[0]
	for name in ("Y", "U", "V"):
		for k, v in x[0].items():
			if k.name == name:
				print("%s = %s" % (k, v))

	"""
	YUV2RGB
	"""

	x = sympy.solve(
	 [e1, e2, e3, e4, e5, e6, e7, e8],
	 exclude=[Y,U,V])
	assert len(x) == 1
	print(x)
	s = x[0]
	for name in ("R", "G", "B"):
		for k, v in x[0].items():
			if k.name == name:
				print("%s = %s" % (k, v))



	"""
	BT.709 changes W_R and W_B:

	- W_R = 0.2126
	- W_B = 0.0722

	"""


	"""
	https://en.wikipedia.org/wiki/Rec._2020

	- W_R = 0.2627
	- W_B = 0.0593

	"""


def do_ycbcr():
	"""
	Perform YCbCr <-> RGB computations.

	The luma and chroma components in YCbCr are calculated from gamma corrected RGB.

	References:

	- https://en.wikipedia.org/wiki/YCbCr

	"""

	"""
	BT.601 defines the following constants:

	- W_R = 0.299
	- W_B = 0.114
	- W_G = 1 - W_R - W_B
	- U_{max} = 0.5
	- V_{max} = 0.5
	- Y = W_R * R + W_G * G + W_B * B
	- U = U_{max} * (B - Y) / (1 - W_B)
	- V = V_{max} * (R - Y) / (1 - W_R)

	"""

	R, G, B, W_R, W_G, W_B, U_max, V_max, Y, U, V \
	 = sympy.symbols("R, G, B, W_R, W_G, W_B, U_max, V_max, Y, U, V")

	e1 = sympy.Eq(W_R, 0.299)
	e2 = sympy.Eq(W_B, 0.114)
	e3 = sympy.Eq(W_G, 1 - W_R - W_B)
	e4 = sympy.Eq(U_max, 0.5)
	e5 = sympy.Eq(V_max, 0.5)
	e6 = sympy.Eq(U, U_max * (B - Y) / (1 - W_B))
	e7 = sympy.Eq(V, V_max * (R - Y) / (1 - W_R))
	e8 = sympy.Eq(Y, R * W_R + G * W_G + B * W_B)

	def process():
		x = sympy.solve([e1, e2, e3], [W_R, W_B, W_G])
		print(x)
		for name in ("W_R", "W_G", "W_B"):
			for k, v in x.items():
				if k.name == name:
					print("%s = %s" % (k, v))

		"""
		RGB2YUV
		"""

		x = sympy.solve(
		 [e1, e2, e3, e4, e5, e6, e7, e8],
		 exclude=[R,G,B])
		assert len(x) == 1
		print(x)
		s = x[0]
		for name in ("Y", "U", "V"):
			for k, v in x[0].items():
				if k.name == name:
					print("%s = %s" % (k, v))

		"""
		YUV2RGB
		"""

		x = sympy.solve(
		 [e1, e2, e3, e4, e5, e6, e7, e8],
		 exclude=[Y,U,V])
		assert len(x) == 1
		print(x)
		s = x[0]
		for name in ("R", "G", "B"):
			for k, v in x[0].items():
				if k.name == name:
					print("%s = %s" % (k, v))


	print("BT.601")
	process()


	"""
	BT.709 changes W_R and W_B:

	- W_R = 0.2126
	- W_B = 0.0722

	"""

	e1 = sympy.Eq(W_R, 0.2126)
	e2 = sympy.Eq(W_B, 0.0722)

	print("BT.709")
	process()


	"""
	https://en.wikipedia.org/wiki/Rec._2020

	- W_R = 0.2627
	- W_B = 0.0593

	"""

	e1 = sympy.Eq(W_R, 0.2627)
	e2 = sympy.Eq(W_B, 0.0593)

	print("Rec.2020")
	process()


def do_lab():
	"""
	https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions

	The division of the domain of the :math:`f` function into two parts
	was done to prevent an infinite slope at :math:`t = 0`.
	The function f was assumed to be linear below some :math:`t = t_0`,
	and was assumed to match the :math:`t^{1/3}` part of the function
	at :math:`t_0` in both value and slope.
	And we have L = 0 for Y = 0.
	"""

	t_0, m, c \
	 = sympy.symbols("t_0, m, c")

	linear_part = m*t_0 + c
	other_part = t_0**sympy.Rational("1/3")

	# equality in value at t_0
	e1 = sympy.Eq(other_part, linear_part)
	print(sympy.pretty(e1))

	# equality in slope
	e2 = sympy.Eq(sympy.diff(e1.lhs, t_0), sympy.diff(e1.rhs, t_0))
	print(sympy.pretty(e2))
	# (if we want to make it solvable without introducing delta)
	#e2 = sympy.Eq(e2.lhs**sympy.Rational("3/2"), e2.rhs**sympy.Rational("3/2"))

	# L=0 at Y=0
	e3 = sympy.Eq(116 * c - 16)
	print(sympy.pretty(e3))

	delta = sympy.Symbol("delta")
	e4 = sympy.Eq(t_0, delta**3)

	x = sympy.solve([e1, e2, e3, e4])
	assert len(x) == 1
	s = x[0]
	print(s)

if __name__ == "__main__":
	do_yuv()
	do_ycbcr()
	do_lab()


