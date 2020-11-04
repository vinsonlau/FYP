from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
import tensorflow as tf
from object_detection.av_detection import detect

class VideoCamera(object):
	def __init__(self):
		# self.video = cv2.VideoCapture(0)
		self.video = cv2.VideoCapture('http://192.168.0.169:8090/camera.mjpeg')

	def __del__(self):
		self.video.release()

	def get_frame(self):
		_, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		image = detect(image)
		_, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()


