import cv2
img = cv2.imread(r'images\alex-shuper-IH82j6ZqtJA-unsplash.jpg', cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
dog = clahe.apply(img)

cv2.imshow('image', dog)
cv2.waitKey(0)
cv2.destroyAllWindows()