import cv2
import numpy as np
import os
import time
import argparse
from deepface import DeepFace
from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from Silent_Face_Anti_Spoofing.src.utility import parse_model_name

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name')

args = parser.parse_args()
model_dir = 'Silent_Face_Anti_Spoofing/resources/anti_spoof_models'
vid = cv2.VideoCapture(0)

model_test = AntiSpoofPredict(0)
image_cropper = CropImage()

while(True):
    ret, image = vid.read()
    image_bbox = model_test.get_bbox(image)
    face_image = image[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
    color = (255, 0, 0)
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        'press c to capture',
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
    cv2.imshow('out',image)
    if cv2.waitKey(25) & 0xFF == ord('c'):
        people_name = args.name
        if people_name is None:
            people_name = '0'
        cv2.imwrite(os.path.join("db_face", people_name + '.jpg'), face_image)
        break