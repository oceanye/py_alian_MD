from py_arduino_motor import MotorControl, MotorControlApp, run_gui

motor_control = MotorControl()
motor_control.init_serial("COM3")  # 请替换为正确的串口名称
motor_control.move_to_position(4, 180)  # 移动第一个电机到90度位置