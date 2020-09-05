import cv2
import numpy as np

cap = cv2.VideoCapture('resources/putt.mov')

# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (15, 15),
maxLevel = 4,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

global point, point_selected



# Mouse function
def select_point(event, x, y, flags, params):
    global old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv2.namedWindow("hi")
cv2.setMouseCallback("hi", select_point)

point = (0,0)
point_selected = True
old_points = np.array([[0, 0]], dtype=np.float32)

available = True
paused = False

available, frame = cap.read()

while available:
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        x, y = new_points.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("hi", frame)

    if paused:
        while True:
            key = cv2.waitKey(5)
            if key != -1:
                print(key)
            if key == 112:
                paused = False
                break
            if key == 27:
                break
    else:
        key = cv2.waitKey(5)
        if key != -1:
            print(key)
        if key == 112:
            paused = True

    if key == 27:
        break

    available, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
