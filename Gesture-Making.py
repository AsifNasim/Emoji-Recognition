import cv2
import numpy as np
import os

image_x, image_y = 50, 50

cap = cv2.VideoCapture(0) #to capture vdo and 0 ndenotes the primary camera
fbag = cv2.createBackgroundSubtractorMOG2() # used Gausian segrentation Algo to subtract the background

def create_folder(folder_name):
    if not os.path.exists(folder_name): # checks for file location
        os.mkdir(folder_name)  #if not exist it creats a new file

def main(g_id):
    total_pics = 1200
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 50, 350, 350

    create_folder("gestures/" + str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0

    while True:
        ret, frame = cap.read() #frame per capture and ret returns the value of each frame capture
        frame = cv2.flip(frame, 1) #flip the image about y-axis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   #convert the color factors from RGB to HSV

        mask2 = cv2.inRange(hsv, np.array([2, 50, 60]), np.array([25, 150, 255])) #
        res = cv2.bitwise_and(frame, frame, mask=mask2) #subtract the background and black out
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) #invert mask is created
        median = cv2.GaussianBlur(gray, (5, 5), 0)  #to remove noise and smoothen the image

        kernel_square = np.ones((5, 5), np.uint8) #matrix of ones and unit8 is range 0 to 255
        dilation = cv2.dilate(median, kernel_square, iterations=2) #merging median and kernel so as to detect the edge of the image
        opening=cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel_square) #outling  the image

        ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY) #segmentation acc to pixel ie. the dark object turs black and the brighter object turns white
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1] #used to detect object

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)  #used to define the countour area
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255)) #print the text onscreen when capturing started
                cv2.imwrite("gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)# saving the image

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", frame)
        cv2.imshow("Thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:
            break


g_id = input("Enter gesture number: ")
main(g_id)