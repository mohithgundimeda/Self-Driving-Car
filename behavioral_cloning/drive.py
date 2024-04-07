import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import os
import numpy as np
from config import *
from load_data import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import tensorflow as tf

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    steering_angle = data["steering_angle"]
    throttle = data["throttle"]
    speed = data["speed"]
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    image_array = cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR)

    image_array = preprocess(frame_bgr=image_array)

    image_array = np.expand_dims(image_array, axis=0)

    steering_angle = float(model.predict(image_array, batch_size=1))

    throttle = 0.28
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':

    from keras.models import model_from_json

    json_path ='pretrained/model.json'
    with open(json_path) as jfile:
        model = model_from_json(jfile.read())

    weights_path = 'pretrained/model.hdf5'
    print('Loading weights: {}'.format(weights_path))
    model.load_weights(weights_path)

    model.compile("adam", "mse")

    app = socketio.Middleware(sio, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
