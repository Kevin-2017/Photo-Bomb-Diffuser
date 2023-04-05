# Goal of this file is to implement pose estimation for inference with Pose2Seg
# based on our previous repo https://github.com/adiojha629/JustDance_Everywhere/blob/main/Others/MoveNet/MoveNet.ipynb

import tensorflow as tf
import numpy as np 
from matplotlib import pyplot as plt
import cv2

if __name__ == "__main__":
    interpreter = tf.lite.Interpreter(
        model_path="movenet_files\lite-model_movenet_singlepose_lightning_3.tflite"
    )
    interpreter.allocate_tensors() # preallocate tensors

    # Pre-inference procedure:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = cv2.imread("movenet_files\\toastmasters_test_image.JPG")
    img = cv2.resize(img,(192,192))
    img = np.expand_dims(img,axis=0) # the added dim is batch size
    input_img = tf.cast(img, dtype=tf.float32)
    # Make a prediction
    interpreter.set_tensor(input_details[0]["index"],np.array(input_img))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
    print(keypoints_with_scores)
