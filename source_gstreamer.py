#!/usr/bin/env python
# -*- coding: utf-8 vi:noet
# Image source using gstreamer

import sys, io, os, re, time, socket, subprocess

import numpy as np

class SourceGstreamer(object):
	def __init__(self, pipeline, like):
		"""
		:param pipeline: pipeline, must end with something named stream
		                 which outputs our video data
		:param like: a numpy array providing the expected buffer geometry
		"""
		self._pipe = os.pipe()
		self._like = like
		self._pipeline = pipeline
		cmd = [
		 "gst-launch-1.0",
		 "--no-fault",
		 #"--messages",
		 #"--tags",
		 #"--index",
		 "--verbose",
		] + pipeline + [
		  "stream.",
		  "!", "fdsink",
		   "fd=%d" % self._pipe[1],
		]
		self._proc = subprocess.Popen(cmd,
		 close_fds=False,
		)
		self._bufsize = self._like.size * self._like.itemsize

	def read(self):
		data = os.read(self._pipe[0], self._bufsize)
		return np.fromstring(data, dtype=self._like.dtype).reshape(self._like.shape)

if __name__ == "__main__":
	import cv2
	pipeline = [
	 "videotestsrc",
	  "is-live=true",
	 "!", "video/x-raw,width=1024,height=1024,format=BGRA,framerate=30/1",
	 "!", "queue",
	  "name=stream",
	]
	src = SourceGstreamer(
	 pipeline=pipeline,
	 like=np.empty((1024, 1024, 4), dtype=np.uint8),
	)
	for idx_frame in range(300):
		print("pouet")
		img = src.read()
		cv2.imwrite("pouet-%03d.png" % idx_frame, img)
