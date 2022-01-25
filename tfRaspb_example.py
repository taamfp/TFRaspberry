### TLite Raspberry Pi example ###
import os
import io 
import time
import cv2
import numpy as np
import matplotlib
import PIL
from picamera.array import PiRGBArray
from picamera import PiCamera
from tflite_runtime.interpreter import Interpreter

model_lite_direct = 'path'
 
def input_tensor_(interpreter, image):
    t_index = inf.get_input_details()[0]['index']
    input_tensor = inf.tensor(t_index)()[0]
    input_tensor[:, :] = image

def cl_image(interpreter, image):
    input_tensor_(interpreter, image)
    
    inf.invoke()
    output_ = inf.get_output_details()[0]
    output = np.squeeze(inf.get_tensor(output_['index']))
    scale, z = output_['quantization']

    return scale

inf = Interpreter(model_lite_direct)
inf.allocate_tensors()

stream = io.BytesIO()

## Read camera

# Initialize camera
camerav2 = PiCamera()

camerav2.resolution = (720, 720)
camerav2.framerate = 40

capt = PiRGBArray(camerav2, (720, 720))

for frame in camerav2.capture_continuous(capt, format='bgr', use_video_port=True):
    image = frame.array
    
    cv2.imshow("camerav2", image)
    
    image_1 = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((inf.get_input_details()[0]['shape'][2], inf.get_input_details()[1]['shape']))
    
    label, classification = cl_image(inf, image_1)

    print('Result: \n', label, np.round(classification*100, 3))
    
    key = cv2.waitKey(1)
    capt.truncate(0)
    
    if key == ord("a"):
        break