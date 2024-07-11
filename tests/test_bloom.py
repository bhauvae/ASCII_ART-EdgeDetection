import cv2
import numpy as np

# read image
img = cv2.imread('testascii.png')

# set arguments
  # threshold to find white
blur_value = 10     # bloom smoothness
gain = 1           # bloom gain in intensity

# convert image to hsv colorspace as floats
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
# h, s, v = cv2.split(hsv)

# Desire low saturation and high brightness for white
# So invert saturation and multiply with brightness
# sv = ((255-s) * v / 255).clip(0,255).astype(np.uint8)

# threshold
# thresh = cv2.threshold(sv, thresh_value, 255, cv2.THRESH_BINARY)[1]

# blur and make 3 channels
blur = cv2.GaussianBlur(img, (0,0), sigmaX=blur_value, sigmaY=blur_value)
# blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# blend blur and image using gain on blur
result = cv2.addWeighted(img, 1, blur, gain, 0)

# save output image
cv2.imwrite('test_bloom.jpg', result)

# display IN and OUT images

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
