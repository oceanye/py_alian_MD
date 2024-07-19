import time

import cv2
import numpy as np
import msvcrt

import camera_utils
from MvCameraControl_class import MvCamera

if __name__ == "__main__":
    cam = camera_utils.open_camera()

    while True:
        current_time = time.time()
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))

        print(formatted_time)

        img = camera_utils.get_frame(cam)

        if img is not None:
            cv2.imshow("Image", img)
        else:
            print("img None")
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()