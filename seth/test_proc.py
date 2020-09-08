import cv2 as cv
import numpy as np
import imutils

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Image Processing Yeah'

def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv.threshold(image, threshold_value, max_binary_value, threshold_type)
    cv.imshow(window_name, dst)
    

max_lowThreshold = 200
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size =  5

def ProcessCannyThreshold(src, threshold=100, kernel_size=5, ratio=ratio):
    low_threshold = threshold
    img_blur = cv.blur(src, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    return dst

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(image, (2,2))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = image * (mask[:,:,None].astype(image.dtype))
    cv.imshow(window_name, mask_highlight(dst, image))

trackbar_field = "Field Trackbar"

def HSV_Filter(val, img):
    filter_field = 2
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV_FULL)
    #new_hsv = np.zeros_like(hsv)
    good_layer = hsv[:,:,filter_field]
    #new_hsv[:,:,filter_field] = hsv[:,:,filter_field]
    new_hsv = np.stack([good_layer, good_layer, good_layer], axis=-1)
    _, filt = cv.threshold(new_hsv, threshold_value, max_binary_value, cv.THRESH_TOZERO)
    mask = filt != 0
    dst = image * mask.astype(image.dtype)
    #dst = (image * (new_hsv/256)).astype(image.dtype)
    cv.imshow(window_name, filt)

def sat_mask(src, threshold, mode=cv.THRESH_TOZERO):
    '''
        `sat_mask` takes an image and masks it so that only areas of the image
        with saturation above `threshold` are kept. All other areas are made full
        black (0,0,0).

        Parameters:
            - src: source image
            - threshold: minimum saturation value to use as cutoff
            - mode: type of threshold to apply, default is keeping all above zero,
                although `cv.THRESH_TOZERO_INV` might also be useful

        Returns:
            - dst: original image with mask applied
    '''
    # transform to HSV color space and extract saturation layer
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV_FULL)
    sat_layer = hsv[:,:,1]
    # duplicate saturation into false 3-channel image for use with cv.threshold
    full_sat = np.stack([sat_layer, sat_layer, sat_layer], axis=-1)
    # perform threshold operation
    _, filt = cv.threshold(full_sat, threshold, 255, mode)
    # create mask based on threshold and apply to original image to "delete" areas below
    # threshodl saturation
    mask = filt != 0
    dst = src * mask.astype(src.dtype)

    return dst

def value_mask(src, threshold, mode=cv.THRESH_TOZERO):
    '''
        `value_mask` takes an image and masks it so that only areas of the image
        with value above `threshold` are kept. All other areas are made full
        black (0,0,0).

        Parameters:
            - src: source image
            - threshold: minimum value value to use as cutoff
            - mode: type of threshold to apply, default is keeping all above zero,
                although `cv.THRESH_TOZERO_INV` might also be useful

        Returns:
            - dst: original image with mask applied
    '''
    # transform to HSV color space and extract value layer
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV_FULL)
    val_layer = hsv[:,:,2]
    # duplicate value into false 3-channel image for use with cv.threshold
    full_val = np.stack([val_layer, val_layer, val_layer], axis=-1)
    # perform threshold operation
    _, filt = cv.threshold(full_val, threshold, 255, mode)
    # create mask based on threshold and apply to original image to "delete" areas below
    # threshold value
    mask = filt != 0
    dst = src * mask.astype(src.dtype)

    return dst

def mask_highlight(masked, original):
    grey_original = cv.cvtColor(cv.cvtColor(original, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
    grey_original //=2
    grey_original[masked != [0,0,0]] = 0

    merged = grey_original + masked
    return merged

def Dual_Process(img, saturation_threshold=0, value_threshold=0):
    '''
        Take an image, apply both a saturation based mask, and value based mask, and display it.
    '''
    sat_processed = sat_mask(img, saturation_threshold)

    val_processed = value_mask(sat_processed, value_threshold)

    grey_original = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

    
    grey_original[val_processed != [0,0,0]] = 0

    merged = grey_original + val_processed

    cv.imshow("Merged", merged)
    cv.imshow("Double Masked", val_processed)

def _DP_Wrap(val):
    Dual_Process(image, 70)


image = cv.imread("pink_putt.jpg")

def main():


    (h,w,d) = image.shape
    print(f"H: {h}, W:{w}, D:{d}")


    # cv.createTrackbar(trackbar_field, window_name, 0, 2, HSV_Filter)
    # cv.createTrackbar(trackbar_type, window_name , 3, max_type, Threshold_Demo)
    # cv.createTrackbar(trackbar_value, window_name , 0, max_value, _DP_Wrap)

    cv.namedWindow(window_name, (cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
    cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    # Create Trackbar to choose Threshold value


    # Call the function to initialize
    '''
    cv.namedWindow("Original", flags=(cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
    cv.imshow("Original", image)'''
    '''

    cv.namedWindow("Merged", flags=(cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
    cv.namedWindow("Double Masked", flags=(cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED))
    Dual_Process(img=image, saturation_threshold=70, value_threshold=70)'''


    # Wait until user finishes program
    cv.waitKey()


