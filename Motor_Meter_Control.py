import time
import serial.tools.list_ports

from py_arduino_motor import MotorControl
from sd76_utils import SD76Device

import serial.tools.list_ports


def find_com_port_by_description(target_description):
    # 获取所有可用的 COM 端口
    ports = serial.tools.list_ports.comports()

    # 遍历所有端口，查找描述中包含目标字符串的端口
    for port in ports:
        if target_description in port.description:
            # 如果找到包含目标字符串的端口，返回该端口名
            return port.device

    # 如果没有找到，返回 None
    return None


# 查找包含 "CH340" 的端口
motor_com_id = find_com_port_by_description("CH340")

# 查找包含 "CP210x" 的端口
meter_com_id = find_com_port_by_description("CP210x")

# 输出结果
if motor_com_id:
    print(f"找到 CH340 设备的 COM 端口: {motor_com_id}")
else:
    print("未找到 CH340 设备。")

if meter_com_id:
    print(f"找到 CP210x 设备的 COM 端口: {meter_com_id}")
else:
    print("未找到 CP210x 设备。")

aim_value = 1000
current_value = 0
# 创建设备实例
device = SD76Device(port=meter_com_id)  # 根据实际情况调整端口


motor_control = MotorControl()
motor_control.init_serial(motor_com_id)

while True:

        current_value = device.read_upper_value() * 1000
        if (aim_value-current_value > 100):
            motor_control.move_to_position(4,0) # 向前
            print("挡位:", "前进")
        elif (aim_value-current_value < -100):
            motor_control.move_to_position(4,180) # 向后
            print("挡位:", "后退")
        else:
            motor_control.move_to_position(4, 90)  # 中档
            print("挡位:", "暂停")

        print("当前值/目标值: ", current_value, "/", aim_value)

        time.sleep(0.2)

