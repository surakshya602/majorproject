import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from docx import Document
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# Initialize predictors
langs = ["ne"]  # Specify language
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

# Streamlit app setup
st.set_page_config(page_title="Nepali OCR App", layout="centered", page_icon="🖋️")

# Header
st.title("📄 Nepali OCR with Editing Features")
st.markdown("Extract Nepali text from an image, edit it, and download the result as a `.docx` file.")

# Sidebar for navigation
st.sidebar.title("EXTRACTION AND RECOGNITION OF NEPALI HANDWRITING USING OCR ")
st.sidebar.header("Workflows")
st.sidebar.markdown("""
Upload an image 📤  
Show the uploaded image 🖼️  
Process the image 🛠️  
Display the result ✨  
Edit text with a Nepali keyboard 📝  
Download the result 📥  
""")
st.sidebar.header("Members: ")
st.sidebar.markdown("""
1. **RAJU CHAPAGAIN (NCE077BCT024)**
2. **RONISH ADHIKARI (NCE077BCT026)**
3. **SANDHYA GIRI (NCE077BCT030)**
4. **SURAKSHYA RANABHAT (NCE077BCT036)**
""")

# Step-by-step buttons
st.header("Uploading Image to perfform OCR")
uploaded_file = st.file_uploader("Upload an image (JPEG, PNG)", type=["jpeg", "jpg", "png"])
def save_to_docx(text, filename="OCRResult.docx"):
    doc = Document()
    doc.add_paragraph(text)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer
if uploaded_file:
    # Show uploaded image
    st.header("Uploaded Image:")
    st.image(uploaded_file, caption="Fig: Raw text image", use_container_width=True)


    # Process image
    if st.button("Start Processing"):
        img = np.array(Image.open(uploaded_file))

        # Apply scan effect
        def scan_effect(img):
            def map(x, in_min, in_max, out_min, out_max):
                return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

            def highPassFilter(img, kSize):
                if not kSize % 2:
                    kSize += 1
                kernel = np.ones((kSize, kSize), np.float32) / (kSize * kSize)
                filtered = cv2.filter2D(img, -1, kernel)
                filtered = img.astype("float32") - filtered.astype("float32")
                filtered = filtered + 127 * np.ones(img.shape, np.uint8)
                filtered = filtered.astype("uint8")
                return filtered

            def blackPointSelect(img, blackPoint):
                img = img.astype("int32")
                img = map(img, blackPoint, 255, 0, 255)
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
                img = img.astype("uint8")
                return img

            def whitePointSelect(img, whitePoint):
                _, img = cv2.threshold(img, whitePoint, 255, cv2.THRESH_TRUNC)
                img = img.astype("int32")
                img = map(img, 0, whitePoint, 0, 255)
                img = img.astype("uint8")
                return img

            def blackAndWhite(img):
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                (l, a, b) = cv2.split(lab)
                img = cv2.add(cv2.subtract(l, b), cv2.subtract(l, a))
                return img

            blackPoint = 66
            whitePoint = 130
            image = highPassFilter(img, kSize=51)
            image_white = whitePointSelect(image, whitePoint)
            img_black = blackPointSelect(image_white, blackPoint)
            image = blackPointSelect(img, blackPoint)
            white = whitePointSelect(image, whitePoint)
            img_black = blackAndWhite(white)
            return img_black

        processed_image = scan_effect(img)
        st.header("Processed(scanned) image :")
        st.image(processed_image, caption="Fig: Scanned image ", use_container_width=True, channels="GRAY")
        

        # OCR Prediction
        processed_pil = Image.fromarray(processed_image)
        predictions = recognition_predictor([processed_pil], [langs], detection_predictor)[0]
        nepali_text = "\n".join([line.text for line in predictions.text_lines])

        # Show OCR result
        st.header("Result")
        st.markdown("### Extracted Nepali Text")
        st.markdown("You can edit by placing cursor")
        edited_text = st.text_area("Edit the Nepali text below:", nepali_text, height=300)

        docx_buffer = save_to_docx(edited_text)
        st.download_button(
        label="📥 Download as DOCX",
        data=docx_buffer,
        file_name="output.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

