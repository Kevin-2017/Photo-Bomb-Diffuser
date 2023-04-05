# Goal of this file is to implement pose estimation for inference with Pose2Seg
# based on our previous repo https://github.com/adiojha629/JustDance_Everywhere/blob/main/Others/MoveNet/MoveNet.ipynb

import tensorflow as tf
import numpy as np 
from matplotlib import pyplot as plt
import cv2

class MoveNet_Predictor():
    def __init__(self):
        # Set up the interpreter
        self.interpreter = tf.lite.Interpreter(
            model_path="movenet_files\lite-model_movenet_singlepose_lightning_3.tflite"
        )
        self.interpreter.allocate_tensors() # preallocate tensors
    def predict(self, input_image):
        '''
        Run Pose Estimation and return the result
        Params:
            input_image (np.array): the image. We don't require that the image is resized correctly
        Returns:
            keypoints_with_scores (3D list): Of size 1x17x3; [x,y,confidence]
        '''
        input_image = self.preprocess(input_image) # we resize and reshape image as necessary

        # Pre-inference procedure
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        # Predict!
        self.interpreter.set_tensor(input_details[0]["index"],input_image)
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])
        return keypoints_with_scores
    def preprocess(self,input_image):
        '''
        Take the image and make sure it has the size [1,192,192,3]
        Param:
            input_image (np.array): input image
        Returns:
            input_image (np.array): input image with size [1,192,192,3]
        '''
        w,h,c = input_image.shape # width, height, channel number
        if w != 192 or h != 192:
            # resize to the correct size
            input_image = cv2.resize(input_image,(192,192))
        if input_image.ndim != 4:
            # Need to add the "batch" dimension.
            # This is where the "1" comes from
            input_image = np.expand_dims(input_image,axis=0)
        assert input_image.shape == (1,192,192,3)
        # Now make sure the type of the numbers is tf.float32
        input_image = tf.cast(input_image, dtype=tf.float32)
        return np.array(input_image)

if __name__ == "__main__":
    # interpreter = tf.lite.Interpreter(
    #     model_path="movenet_files\lite-model_movenet_singlepose_lightning_3.tflite"
    # )
    # interpreter.allocate_tensors() # preallocate tensors

    # Pre-inference procedure:
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    img = cv2.imread("movenet_files\\toastmasters_test_image.JPG")
    # img = cv2.resize(img,(192,192))
    # img = np.expand_dims(img,axis=0) # the added dim is batch size
    # print(img.shape)
    # input_img = tf.cast(img, dtype=tf.float32)
    # print(type(input_img))
    # # Make a prediction
    # interpreter.set_tensor(input_details[0]["index"],np.array(input_img))
    # interpreter.invoke()
    # keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])

    mdl = MoveNet_Predictor() # init movenet model
    keypoints_with_scores = mdl.predict(img) # make predictions
    print(keypoints_with_scores)
