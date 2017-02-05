#!/usr/bin/env python
# -*- coding: utf-8 vi:noet

import sympy


def do_yuv():
	"""
	Perform YUV <-> RGB computations.

	The luma and chroma components in YCbCr are calculated from gamma corrected RGB.
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
	- W_G = 0.6780
	- W_B = 0.0593


	"""

if __name__ == "__main__":
	do_yuv()
