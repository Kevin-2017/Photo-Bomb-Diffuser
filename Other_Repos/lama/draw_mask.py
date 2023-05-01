'''
purpose of this file is to allow users to draw a mask over an img
'''
import cv2
import numpy as np

draw_x, draw_y = None,None
drawing_on = False
def click_event(event,x,y,flags,params):
    '''
    This function is called whenever a button is pressed!
    x,y are the locations of the button click
    '''
    global draw_x, draw_y, drawing_on
    if event == cv2.EVENT_LBUTTONDOWN: # only draw after odd number of left button clicks
        drawing_on = not drawing_on

    if event == cv2.EVENT_MOUSEMOVE: # if Right mouse button clicked
        if drawing_on:
            # display x,y
            print(x,",",y)
            draw_x, draw_y = x, y

if __name__ == "__main__":
    black_board = np.full(shape=(512,512),
                          fill_value=0,
                          dtype=float) # start with all black
    img = cv2.imread("testing_imgs/graduation.png")
    img = cv2.resize(img,(512,512))
    FILLED = -1 # means filled shape in opencv
    # a click is detected.
    while True:
        if draw_y != None and draw_x != None:
            black_board = cv2.circle(black_board,
                                     center=(draw_x,draw_y),
                                     radius=10,
                                     color=(255,255,255),
                                     thickness=FILLED) # draw white circle
            img = cv2.circle(img,
                            center=(draw_x,draw_y),
                            radius=10,
                            color=(255,255,255),
                            thickness=FILLED) # draw white circle
        cv2.imshow("Mask",black_board)
        cv2.imshow("Image",img)
        cv2.setMouseCallback("Image",click_event) # set the function to run when
        
        key = cv2.waitKey(1) # collect key
        if key == 115 or key == 83: # either s or S for save
            cv2.imwrite("testing_masks/drawn_mask.jpg",black_board)
            print("Saved image!")
            break
        if key == 27: # if ESC we exit
            break
    cv2.destroyAllWindows()