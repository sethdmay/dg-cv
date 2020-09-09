import numpy as np
import cv2 as cv

cap = cv.VideoCapture("resources/putt.mov")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

cv.namedWindow('frame2', (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
cv.namedWindow('original', (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))

while ret:
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])


    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    #bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    
    norm_mag = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    highlight = cv.cvtColor(cv.cvtColor(frame2, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    highlight[:,:,2] = np.maximum(norm_mag,highlight[:,:,2])

    cv.imshow("original",highlight)
    #cv.imshow('frame2',bgr)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('out/opticalfb.png',frame2)
        cv.imwrite('out/opticalhsv.png',bgr)
    prvs = next