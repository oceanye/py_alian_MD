# main.py

import cv2
import numpy as np
import msvcrt

import camera_utils

if __name__ == "__main__":
    cam = camera_utils.open_camera()

    while True:
        data, width, height = camera_utils.get_frame(cam)

        if data is not None:
            img = np.frombuffer(data, dtype=np.uint8)
            img = img.reshape((height, width, 1))

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if msvcrt.kbhit():
            break

    camera_utils.close_camera(cam)
    cv2.destroyAllWindows()