import numpy as np
import cv2
import matplotlib.pyplot as plt


ORIGINAL_WINDOW_TITLE = 'Original'
FIRST_FRAME_WINDOW_TITLE = 'First Frame'
DIFFERENCE_WINDOW_TITLE = 'Difference'


canvas = None
drawing = False # true if mouse is pressed

#Retrieve first frame
def initialize_camera(cap):
    _, frame = cap.read()
    return frame


# mouse callback function
def mouse_draw_rect(event,x,y,flags, params):
    global drawing, canvas

    if drawing:
        canvas = params[0].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        params.append((x,y)) #Save first point

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(canvas, params[1],(x,y),(0,255,0),2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        params.append((x,y)) #Save second point
        cv2.rectangle(canvas,params[1],params[2],(0,255,0),2)


def select_roi(frame):
    global canvas
    canvas = frame.copy()
    params = [frame]
    ROI_SELECTION_WINDOW = 'Select ROI'
    cv2.namedWindow(ROI_SELECTION_WINDOW)
    cv2.setMouseCallback(ROI_SELECTION_WINDOW, mouse_draw_rect, params)
    roi_selected = False
    while True:
        cv2.imshow(ROI_SELECTION_WINDOW, canvas)
        key = cv2.waitKey(10)

        #Press Enter to break the loop
        if key == 13:
            break;


    cv2.destroyWindow(ROI_SELECTION_WINDOW)
    roi_selected = (3 == len(params))

    if roi_selected:
        p1 = params[1]
        p2 = params[2]
        if (p1[0] == p2[0]) and (p1[1] == p2[1]):
            roi_selected = False

    #Use whole frame if ROI has not been selected
    if not roi_selected:
        print('ROI Not Selected. Using Full Frame')
        p1 = (0,0)
        p2 = (frame.shape[1] - 1, frame.shape[0] -1)


    return roi_selected, p1, p2




if __name__ == '__main__':

    cap = cv2.VideoCapture('resources/putt69.m4v')

    #Grab first frame
    first_frame = initialize_camera(cap)

    #Select ROI for processing. Hit Enter after drawing the rectangle to finalize selection
    roi_selected, point1, point2 = select_roi(first_frame)

    #Grab ROI of first frame
    first_frame_roi = first_frame[point1[1]:point2[1], point1[0]:point2[0], :]

    #An empty image of full size just for visualization of difference
    difference_image_canvas = np.zeros_like(first_frame)

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:

            #ROI of current frame
            roi = frame[point1[1]:point2[1], point1[0]:point2[0], :]

            difference = cv2.absdiff(first_frame_roi, roi)
            difference = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
            # difference = cv2.GaussianBlur(difference, (15, 15), 0)
            median = cv2.medianBlur(difference, 15)

            _, difference = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)

            dilate_image = cv2.erode(difference, None, iterations=1) #ben

            image, contours, hierachy = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 150:  # if contour area is less then 800 non-zero(not-black) pixels(white)
                    (x, y, w, h) = cv2.boundingRect(c)  # x,y are the top left of the contour and w,h are the width and hieght

                    cv2.rectangle(frame, (x, y), (x+point1[0], y+point1[1]), (255, 0, 0), 2)

                else:
                    pass

            cv2.imshow(ORIGINAL_WINDOW_TITLE, frame)



            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
