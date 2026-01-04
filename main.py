import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("best.pt")
capture = cv2.imread("test.jpg")



results = model(capture)[0]

if results.masks is not None and results.masks.xy is not None:
    for polygon in results.masks.xy:
        contour = np.array(polygon, dtype=np.int32)
        area = cv2.contourArea(contour)
    print(polygon)

    result_img = results.plot()
    cv2.imwrite("result.jpg",result_img)
    with open("log.txt",'w') as f:
        f.write(repr(results))
    with open("coords.txt", 'w') as f:
        f.write(repr(results.masks))

else:
    result_img = capture

capture.release()