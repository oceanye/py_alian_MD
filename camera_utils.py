# camera_utils.py

import sys
from ctypes import *
import cv2
import numpy as np

sys.path.append("../MvImport")
from MvCameraControl_class import *


def open_camera():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    return cam


def close_camera(cam):
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()


def get_frame_black(cam):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if None != stOutFrame.pBufAddr and 0 == ret:
        print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
        stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))

        data = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
        cdll.msvcrt.memcpy(byref(data), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)

        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)

        return data, stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight
    else:
        print("no data[0x%x]" % ret)
        return None, None, None


def get_frame(cam):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if None != stOutFrame.pBufAddr and 0 == ret:
        print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
        stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))

        data = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
        cdll.msvcrt.memcpy(byref(data), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)

        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)

        # Convert the image from YUV422 to BGR format
        if stOutFrame.stFrameInfo.enPixelType == 17301513:  # YUV422_Packed
            img = np.frombuffer(data, dtype=np.uint8)
            img = img.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, 2))
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
        else:
            print("unsupported pixel type: %d" % stOutFrame.stFrameInfo.enPixelType)
            return None

        return img
    else:
        print("no data[0x%x]" % ret)
        return None