import socketio
import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 35


def img_preprocess(img):
    #Crop the image
    img = img[60:135, :, :]
    # Convert color to yuv y-brightness, u,v chrominants(color)
    # Recommend in the NVIDIA paper
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # Apply Gaussian Blur
    # As suggested by NVIDIA paper
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    speed = float(data['speed'])
    throttle = 1.0 - speed/speed_limit
    steering_angle = float(model.predict(image))
    send_control(steering_angle, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle' : steering_angle.__str__(),
        'throttle' : throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)