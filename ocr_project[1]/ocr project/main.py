from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import time
import json
import os
from jiwer import cer,wer
t1 = time.time()
import cv2
import numpy as np
# image = Image.open("aa2d3957-1426.png")
langs = ["ne"] # Replace with your languages or pass None (recommended to use None)
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()
def map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def highPassFilter(img,kSize):
    if not kSize%2:
        kSize +=1
    kernel = np.ones((kSize,kSize),np.float32)/(kSize*kSize)
    filtered = cv2.filter2D(img,-1,kernel)
    filtered = img.astype('float32') - filtered.astype('float32')
    filtered = filtered + 127*np.ones(img.shape, np.uint8)
    filtered = filtered.astype('uint8')
    return filtered

def blackPointSelect(img, blackPoint):
    img = img.astype('int32')
    img = map(img, blackPoint, 255, 0, 255)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
    img = img.astype('uint8')
    return img

def whitePointSelect(img,whitePoint):
    _,img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)
    img = img.astype('int32')
    img = map(img, 0, whitePoint, 0, 255)
    img = img.astype('uint8')
    return img

def blackAndWhite(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    (l,a,b) = cv2.split(lab)
    img = cv2.add( cv2.subtract(l,b), cv2.subtract(l,a) )
    return img
def scan_effect(img):
    blackPoint = 66
    whitePoint = 130
    image = highPassFilter(img,kSize = 51)
    image_white = whitePointSelect(image, whitePoint)
    img_black = blackPointSelect(image_white, blackPoint)
    image=blackPointSelect(img,blackPoint)
    white = whitePointSelect(image,whitePoint)
    img_black = blackAndWhite(white)
    return img_black
def process_single_image(image_path, ground_truth_text):
    img = cv2.imread(image_path)
    image = scan_effect(img)
    pil_image_image = Image.fromarray(image)
    # image = Image.open(image_path)
    predictions = recognition_predictor([pil_image_image], [langs], detection_predictor)[0]
    total_text = ""
    text = [line.text for line in predictions.text_lines]
    total_text+= "".join(text)
    predicted_list = total_text.split(" ")
    predicted_text = " ".join(predicted_list)
    return wer(ground_truth_text, predicted_text)

def calculate_average_wer_parallel(image_folder, ground_truth_file):
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    wer_list = []
    filenames = []
    for file_name in os.listdir(image_folder):
        filenames.append(file_name)
    for i in range(len(ground_truth)):
        image_name = ground_truth[i]['ocr'].split("/")[-1]
        ground_text = "".join(ground_truth[i]['transcription'])
        for j in range(len(filenames)):
            if image_name == filenames[j]:
                image_paths = os.path.join(image_folder,image_name)
                wer_list.append(process_single_image(image_paths,ground_text))
    return wer_list
# Example Usage
image_folder_path = "./OCR/data/raju 2/images"
ground_truth_json_path = "./OCR/data/raju 2/project-1-at-2025-01-20-13-01-1272a784.json"
average_wer = calculate_average_wer_parallel(image_folder_path, ground_truth_json_path)
print(f"Average WER (Parallel): {average_wer}")
# Average WER (Parallel): [40.666666666666664, 132.0, 105.0, 18.666666666666668, 43.5, 72.0, 90.0, 101.0, 75.0, 52.0, 23.0, 35.5, 12.5, 57.0, 13.4, 33.25, 19.6, 7.5, 19.8, 9.5]

