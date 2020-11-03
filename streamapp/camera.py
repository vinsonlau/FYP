from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
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
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.
		image = detect(image)
		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()


