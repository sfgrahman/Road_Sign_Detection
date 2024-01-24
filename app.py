import streamlit as st
from ultralytics import YOLO 
import numpy as np
import cv2
from PIL import Image

model = YOLO('best.pt')

# function to convert file buffer to cv2 image
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)



st.set_page_config(
  page_title="Road Sign Detection Using YOLO8",
  page_icon="🚀"
)

st.title('Road Sign Detection Using YOLO8')

st.markdown("""
            <p style="color:#0FFF00;font-size:20px;">
            This is an application for road sign detection using YOLO. It will detect 4 classes</br>
            - Trafic Light<br>
- Stop</br>
- Speedlimit</br>
- Crosswalk</br>
            </p>
            """, unsafe_allow_html=True)

img_files = st.file_uploader(label="Choose an image files",
                 type=['png', 'jpg', 'jpeg'],
                 accept_multiple_files=True)

for n, img_file_buffer in enumerate(img_files):
  if img_file_buffer is not None:
    # 1) image file buffer will converted to cv2 image
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)
    # 2) pass image to the model to get the detection result
    results = model.predict(open_cv_image,  conf=0.30) 
    # 3) show result image using st.image()
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        st.image(im, channels="BGR", \
  caption=f'Detection Results ({n+1}/{len(img_files)})')
    #pass

st.markdown("""
  <p style='text-align: center; font-size:16px; margin-top: 32px'>
   MoududurShamim @2024
  </p>
""", unsafe_allow_html=True)