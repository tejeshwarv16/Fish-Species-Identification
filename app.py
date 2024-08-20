import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("C:/Users/tejes/Downloads/my_model2.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0:'Anthias anthias',
            1:'Atherinomorus lacunosus',
            2:'Belone belone',
            3:'Boops boops',
            4:'Chlorophthalmus agassizi',
            5:'Coris julis',
            6:'Dasyatis centroura',
            7:'Epinephelus caninus',
            8:'Gobius niger',
            9:'Mugil cephalus',
            10:'Phycis phycis',
            11:'Polyprion americanus',
            12:'Pseudocaranx dentex',
            13:'Rhinobatos cemiculus',
            14:'Scomber japonicus',
            15:'Solea solea',
            16:'Squalus acanthias',
            17:'Tetrapturus belone',
            18:'Trachinus draco',
            19:'Trigloporus lastoviza'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(200,200))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
