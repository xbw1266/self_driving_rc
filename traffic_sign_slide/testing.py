from traffic_detect import traffic_sign_detect
import cv2

image_dir = '../../'
model_name = 'german_model.h5'

test = traffic_sign_detect(model_name)
cap = cv2.VideoCapture(0)
ret, img = cap.read()

while ret is True:
    ret, img = cap.read()
    test.loadimage(img)
    test.find_box()
    test.draw_heatmap()

cap.release()
cv2.destroyAllWindow()
