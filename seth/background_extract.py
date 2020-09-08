import cv2 as cv
import numpy as np


window_name = "FG Mask"
paused = False

# Lucas kanade params
lk_params = dict(winSize = (15, 15),
maxLevel = 4,
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))



video = cv.VideoCapture('resources/putt.mov')
bg_extract = cv.createBackgroundSubtractorKNN(history=750, dist2Threshold=800)
bg_extract.setNSamples(10)

available, frame = video.read()


cv.namedWindow(window_name, (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
cv.namedWindow("Original", (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))


def pause(*args):
    global paused
    paused = not paused

cv.createButton("Pause", pause, None, cv.QT_PUSH_BUTTON)

point_selected = False
point = ()
old_points = np.array([[]])


# Mouse function
def select_point(event, x, y, flags, params):
    global old_points, point, point_selected
    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

def mask_highlight(mask, original):
    pure_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    pure_mask = (pure_mask == 255)
    
    grey_original = cv.cvtColor(cv.cvtColor(original, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    grey_original //= 10
    grey_original[pure_mask != [0,0,0]] = 0

    masked = original * pure_mask.astype(original.dtype)

    merged = grey_original + masked
    return merged, masked

cv.setMouseCallback(window_name, select_point)

while available:
    fg_mask = bg_extract.apply(frame)

    fg_highlight, fg_alone = mask_highlight(fg_mask, frame)

    gray_frame = cv.cvtColor(fg_highlight, cv.COLOR_BGR2GRAY)
    

    if point_selected:
        cv.circle(fg_highlight, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        
        old_points = new_points

        x, y = new_points.ravel()
        cv.circle(fg_highlight, (x, y), 5, (0, 255, 0), -1)

    old_gray = gray_frame.copy()

    cv.imshow(window_name, fg_highlight)
    cv.imshow("Original", frame)
    key = cv.waitKey(2)

    while True:
        if paused:
            key = cv.waitKey(100)
            if key == 112:
                paused = False
                break
            if key == 27:
                break
        else:
            key = cv.waitKey(2)
            break

    if key == 27:
        break

    available, frame = video.read()

video.release()
cv.destroyAllWindows()