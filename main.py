import cv2
import numpy as np
import os
import time
from deepface import DeepFace
from Silent_Face_Anti_Spoofing.src.anti_spoof_predict import AntiSpoofPredict
from Silent_Face_Anti_Spoofing.src.generate_patches import CropImage
from Silent_Face_Anti_Spoofing.src.utility import parse_model_name

def run():

    model_dir = 'Silent_Face_Anti_Spoofing/resources/anti_spoof_models'
    vid = cv2.VideoCapture(0)

    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    if os.path.isfile(os.path.join("db_face", "representations_arcface.pkl")):
        os.remove(os.path.join("db_face", "representations_arcface.pkl"))

    while(True):
        ret, image = vid.read()
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            face_image = image[image_bbox[1]:image_bbox[1]+image_bbox[3], image_bbox[0]:image_bbox[0]+image_bbox[2]]
            if face_image is None:
                continue
            df = DeepFace.find(face_image,
                    db_path = "db_face", 
                    model_name = 'ArcFace',
                    distance_metric= 'cosine',
                    enforce_detection= False,
                    detector_backend= 'skip'
                )
            if len(df['identity']) != 0 and df['ArcFace_cosine'][0] < 0.4:
                person_name = os.path.basename(df['identity'][0]).split('.')[0]
                result_text = "RealFace: " + person_name
                color = (255, 0, 0)
            else:
                person_name = 'no indentity'
                result_text = "RealFace: " + person_name
                color = (0, 0, 255)
        else:
            result_text = "FakeFace"
            color = (0, 0, 255)
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
        cv2.imshow('out',image)
        cv2.waitKey(25)

run()