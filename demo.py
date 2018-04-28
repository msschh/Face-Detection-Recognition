"""
Demo for the face detection. Runs the sliding window detector on demo.jpg
Displays the input image, the localized faces and the sliding window mask

"""

import FaceFinder
import cv2

model_path = 'face_model'
img = cv2.imread("demo.jpg",0)
face_img = FaceFinder.localize(img, model_path)
cv2.imshow("Face", face_img)
cv2.waitKey(0)
cv2.destroyAllWindows()