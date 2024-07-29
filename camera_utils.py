from MvCameraControl_class import *
import cv2
import numpy as np
import threading
import random
import time
import inspect


def open_camera():
    # 枚举设备
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, device_list)
    if ret != 0:
        print("Enum devices fail! ret[0x%x]" % ret)
        return None

    # 选择第一个设备并创建句柄
    if device_list.nDeviceNum == 0:
        print("Find no device!")
        return None
    print("Find %d devices!" % device_list.nDeviceNum)

    camera = MvCamera()
    stDeviceList = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = camera.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("Create handle fail! ret[0x%x]" % ret)
        return None

    # 打开设备
    ret = camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Open device fail! ret[0x%x]" % ret)
        return None

    return camera


def get_frame(camera):
    # 开始取流
    ret = camera.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing fail! ret[0x%x]" % ret)
        return None

    # 获取一帧图像
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    pData = (c_ubyte * (sizeof(c_ubyte) * 1024 * 1024 * 10))()
    ret = camera.MV_CC_GetOneFrameTimeout(pData, sizeof(pData), stFrameInfo, 1000)
    if ret == 0:
        print("Get one frame: Width[%d], Height[%d], PixelType[0x%x], FrameNum[%d]" %
              (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.enPixelType, stFrameInfo.nFrameNum))

        # 解析Bayer格式图像
        if stFrameInfo.enPixelType == 0x1080009:  # PIXEL_FORMAT_BAYER_RG8
            data_size = stFrameInfo.nWidth * stFrameInfo.nHeight
            image_data = bytes(pData)[:data_size]
            image = np.frombuffer(image_data, dtype=np.uint8).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
            image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
        else:
            print("Unsupported pixel format: 0x%x" % stFrameInfo.enPixelType)
            image = None
    else:
        print("No data, ret[0x%x]" % ret)
        image = None

    # 停止取流
    ret = camera.MV_CC_StopGrabbing()
    if ret != 0:
        print("Stop grabbing fail! ret[0x%x]" % ret)
        return None


    return image

