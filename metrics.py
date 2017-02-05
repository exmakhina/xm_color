#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# “Metrics” for comparing image-like stuff


import numpy as np
import cv2

def CIE_deltaE_2000(lab1, lab2, kl=1.0, kc=1.0, kh=1.0):
	"""
	Compute the CIEDE2000 color difference function on 2 inputs

	:param lab1,lab2: input values/images in LAB color space
	:param kl: compensation for lightness
	:param kc: compensation for chroma
	:param kh: compensation for hue

	See:

	- https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
	- http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
	"""
	L1, a1, b1 = cv2.split(lab1)
	L2, a2, b2 = cv2.split(lab2)
	c1 = np.sqrt(a1**2+b1**2)
	c2 = np.sqrt(a2**2+b2**2)
	cm = (c1+c2)/2
	G = 0.5 * (1 - np.sqrt((cm**7)/(cm**7 + 25**7)))
	ap1 = (1+G)*a1
	ap2 = (1+G)*a2
	Cp2 = np.sqrt(ap2**2+b2**2)
	Cp1 = np.sqrt(ap1**2+b1**2)

	Cpprod = Cp1 * Cp2
	zcidx = Cpprod == 0

	hp1 = np.arctan2(b1, ap1)
	hp1[hp1<0] += 2*np.pi
	hp1[np.abs(ap1)+np.abs(b1) == 0] = 0
	hp2 = np.arctan2(b2, ap2)
	hp2[hp2<0] += 2*np.pi
	hp2[np.abs(ap2)+np.abs(b2) == 0] = 0

	dL = L2 - L1
	dC = Cp2 - Cp1
	dhp = hp2 - hp1
	dhp[dhp>np.pi] -= 2*np.pi
	dhp[dhp<-np.pi] += 2*np.pi
	dhp[zcidx] = 0
	dH = 2 * np.sqrt(Cpprod) * np.sin(dhp/2)

	Lp = (L1+L2)/2
	Cp = (Cp1+Cp2)/2

	hp = (hp1+hp2)/2
	hp[hp2-hp1 > 2*np.pi] -= 2*np.pi
	hp[hp2-hp1 < -2*np.pi] += 2*np.pi
	hp[zcidx] = hp2[zcidx] + hp1[zcidx]

	Lpm502 = (Lp-50)**2

	dtheta = (np.pi/6)*np.exp(- ((180/np.pi*hp-275)/25)**2)
	Rc = 2 * ((Cp**7)/(Cp**7 + 25**7))**0.5;

	Sl = 1 + 0.015*Lpm502/(20+Lpm502)**0.5
	Sc = 1 + 0.045*Cp

	T = 1 \
	 - 0.17*np.cos(hp - np.pi/6) \
	 + 0.24*np.cos(2*hp) \
	 + 0.32*np.cos(3*hp+np.pi/30) \
	 - 0.20*np.cos(4*hp-63*np.pi/180)

	Sh = 1 + 0.015*Cp*T
	RT = - np.sin(2*dtheta)*Rc

	klSl = kl*Sl
	kcSc = kc*Sc
	khSh = kh*Sh

	dE00 = np.sqrt( (dL/klSl)**2 + (dC/kcSc)**2 + (dH/khSh)**2 + RT*(dC/kcSc)*(dH/khSh))

	return dE00
