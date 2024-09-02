from datetime import time

from py_arduino_motor import MotorControl
from sd76_utils import SD76Device


aim_value = 3800
current_value = 0
# 创建设备实例
device = SD76Device(port='COM3')  # 根据实际情况调整端口



motor_control = MotorControl()
motor_control.init_serial("COM3")

while abs(aim_value-current_value)>50:

    # 读取上排显示值
    current_value = device.read_upper_value()*1000

    print(f"现位置: {current_value}")

    if (aim_value-current_value > 50):
        motor_control.move_to_position(4,0) # 向前
    else:
        motor_control.move_to_position(4,180) # 向后
    time.sleep(0.5)

motor_control.move_to_position(4,90) # 向前