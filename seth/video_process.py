import numpy as np
import cv2 as cv

import test_proc

def compare_mask(masked, original, color=0):
    grey_original = cv.cvtColor(cv.cvtColor(original, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    grey_original //=2
    grey_original[masked != [0,0,0]] = 0

    high_masked = np.zeros_like(masked)
    high_masked[:,:,color] = masked[:,:,color]

    merged = grey_original + high_masked
    return high_masked

wn1 = "125"
wn2 = "90"

ben_throw = cv.VideoCapture('IMG_2555.MOV')


frame_available, frame = ben_throw.read()

cv.namedWindow(wn1, (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
cv.namedWindow(wn2, (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))

while frame_available:
    edged1 = test_proc.ProcessCannyThreshold(frame, threshold=100, kernel_size=4, ratio=1.5)
    mask_and_shade1 = test_proc.mask_highlight(edged1, frame)

    #edged2 = test_proc.ProcessCannyThreshold(frame, threshold=90, kernel_size=4, ratio=1.5)
    #mask_and_shade2 = test_proc.mask_highlight(edged2, frame)

    #grey_original = cv.cvtColor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    #grey_original //=2

    #summed = grey_original + mask_and_shade1 + mask_and_shade2
    #summed = np.min(np.stack(summed,255], 255)

    cv.imshow(wn1, mask_and_shade1)
    #cv.imshow(wn2, mask_and_shade2)
    cv.waitKey(2)



    frame_available, frame = ben_throw.read()

