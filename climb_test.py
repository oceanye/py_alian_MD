from py_arduino_motor import MotorControl, MotorControlApp, run_gui
import serial.tools.list_ports
import time

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

motor_com_id = find_com_port_by_description("CH340")

try:
    motor_control = MotorControl()
    motor_control.init_serial(motor_com_id)  # 请替换为正确的串口名称
    motor_connect = True
except Exception as e:
    print("motor not connect with COM3")



#12 - 上主轴
# 12，0， 上-内
#12，90 上-中
#12，180 上-外
#13 - 下主轴

# 14，0 上张开
# 14, 90 上和龙
# 15，0 下张开
#15，90 下和龙

motor_control.move_to_position(12, 90)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(13, 90)  # 移动第一个电机到90度位置
time.sleep(1)

motor_control.move_to_position(14, 0)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(15, 0)  # 移动第一个电机到90度位置
time.sleep(1)

#-----------

motor_control.move_to_position(12, 0)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(13, 00)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(14, 90)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(15, 100)  # 移动第一个电机到90度位置
time.sleep(1)

#-----锁紧

#---松开甩臂

motor_control.move_to_position(15, 0)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(13, 90)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(12, 45)  # 移动第一个电机到90度位置
time.sleep(0.2)
motor_control.move_to_position(12, 90)  # 移动第一个电机到90度位置
time.sleep(0.2)
motor_control.move_to_position(12, 135)  # 移动第一个电机到90度位置
time.sleep(0.2)
motor_control.move_to_position(12, 180)  # 移动第一个电机到90度位置
time.sleep(1)

motor_control.move_to_position(13, 170)  # 移动第一个电机到90度位置90)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(15, 90)  # 移动第一个电机到90度位置90)  # 移动第一个电机到90度位置
time.sleep(1)


#-----



motor_control.move_to_position(14, 0)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(12, 90)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(13, 45)  # 移动第一个电机到90度位置
time.sleep(0.2)
motor_control.move_to_position(13, 90)  # 移动第一个电机到90度位置
time.sleep(0.2)
motor_control.move_to_position(13, 135)  # 移动第一个电机到90度位置
time.sleep(0.2)
motor_control.move_to_position(13, 180)  # 移动第一个电机到90度位置
time.sleep(1)

motor_control.move_to_position(12, 0)  # 移动第一个电机到90度位置90)  # 移动第一个电机到90度位置
time.sleep(1)
motor_control.move_to_position(14, 90)  # 移动第一个电机到90度位置90)  # 移动第一个电机到90度位置
time.sleep(1)
