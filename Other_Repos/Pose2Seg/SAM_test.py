from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import cv2
import numpy as np
if __name__ == "__main__":
    img = cv2.imread("movenet_files\\toastmasters_test_image.JPG")
    # predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    # predictor.set_image(img)
    # masks,_,_ = predictor.predict(np.array([[500,500]]),np.array([1]))
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    masks = mask_generator.generate(img)
    # print("Mask shape is ",masks.shape)
    # print("Mask type is ",type(masks))
    # # user = input("Code: ")
    # # while user != "n":
    # #     eval(user)
    # #     user = input("Code: ")
    # cv2.imshow("Mask",masks.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()