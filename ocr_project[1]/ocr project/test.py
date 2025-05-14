# predictions = recognition_predictor([image], [langs], detection_predictor)[0]
# total_text = ""
# text = [line.text for line in predictions.text_lines]
# total_text+= "".join(text)
# predicted_list = total_text.split(" ")
# print(f"time taken to process oe image : {time.time()-t1} seconds")
# print(predicted_list)

# ground_truth = [
#       "आज",
#       "हुन",
#       "लाएको",
#       "यस",
#       "वादविवाद",
#       "प्रतियोगितामा",
#       "मलाई",
#       "पनि",
#       "बोल्ने",
#       "अवसर",
#       "जुराइदिएकोमा",
#       "आयोजक",
#       "समितिलाई",
#       "धन्यवाद",
#       "दिन",
#       "चाहान्छु",
#       "।",
#       "म",
#       "आज",
#       "धनभन्दा",
#       "मन",
#       "ठुलो",
#       "भन्ने",
#       "विषयमा",
#       "बिपक्ष्",
#       "बाट",
#       "बोल्न",
#       "चाहान्छु",
#       "महोदय,",
#       "हुन",
#       "त",
#       "हो",
#       "घनभन्दा",
#       "मन",
#       "ठुलो",
#       "हुन्छ",
#       "।",
#       "किनभने",
#       "मन",
#       "भएका",
#       "मानिस",
#       "एकदमै",
#       "राम्रो",
#       "र",
#       "दयालु",
#       "हुन्छ ।",
#       "धनमात्र",
#       "भएर",
#       "के",
#       "गर्नु",
#       "?",
#       "त्यो",
#       "धन",
#       "कमाउन",
#       "मन",
#       "को",
#       "आवश्यक",
#       "हुन्छ",
#       "।",
#       "यदि",
#       "मन",
#       "छ",
#       "त",
#       "संसारभरि",
#       "शान्ति",
#       "छ",
#       "।",
#       "धन",
#       "मात्र",
#       "भएर",
#       "संसार",
#       "चल्दैन",
#       "।",
#       " मानिसको",
#       "मन",
#       "सङ्‌कुचित",
#       "भएमा",
#       "उसले",
#       "धन",
#       "भए",
#       "पनि",
#       "जीवनमा",
#       "केहि",
#       "गर्न",
#       "सक्दैना",
#       "मानिसको",
#       "जीवनमा",
#       "केहि",
#       "सफलता",
#       "हासिल",
#       "गर्न",
#       "पनि",
#       "मनमा",
#       "उदारता",
#       "र",
#       "भावनामा",
#       "फराकिलोपन",
#       "हुनै",
#       "पर्छ",
#       "।",
#       "धनमात्र",
#       "भएर",
#       "मानिस",
#       "बाँच्न",
#       "सक्दैन",
#       "मानिसलाई",
#       "धनको",
#       "प्राप्त",
#       "गर्न",
#       "मन",
#       "चाहिन्छ",
#       "।",
#       "महाकवि",
#       "लक्ष्मी",
#       "प्रसाद",
#       "देवकोटाले"
#     ]
# predicted_text = " ".join(predicted_list)
# ground_truth_text = " ".join(ground_truth)
# # Calculate WER
# wer_score = wer(ground_truth_text, predicted_text)
# print(f"WER: {wer_score:.4f}")
import numpy as np
import cv2
from PIL import Image

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
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
langs = ["ne"] # Replace with your languages or pass None (recommended to use None)
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()
from PIL import Image
img = cv2.imread("./OCR/data/raju 1/images/dfc1289f-1361.png")
image = scan_effect(img)
image = Image.fromarray(image)
predictions = recognition_predictor([image], [langs], detection_predictor)[0]
total_text = ""
text = [line.text for line in predictions.text_lines]
print(text)
