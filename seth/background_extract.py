import cv2 as cv
import numpy as np


window_name = "FG Mask"
paused = False

video = cv.VideoCapture('resources/drive-br.mov')
bg_extract = cv.createBackgroundSubtractorKNN(history=750, dist2Threshold=800)
bg_extract.setNSamples(10)


available, frame = video.read()


cv.namedWindow(window_name, (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
cv.namedWindow("Original", (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))


def pause(*args):
    global paused
    paused = not paused

cv.setMouseCallback(window_name, pause)

while available:
    fg_mask = bg_extract.apply(frame)

    cv.imshow(window_name, fg_mask)
    cv.imshow("Original", frame)
    key = cv.waitKey(1)

    if paused:
        while True:
            key = cv.waitKey(100)
            if key == 112:
                paused = False
                break
            if key == 27:
                break
    else:
        key = cv.waitKey(2)

    if key == 27:
        break

    available, frame = video.read()

