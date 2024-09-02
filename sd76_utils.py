import serial
import struct
import time

def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return struct.pack('<H', crc)

def read_register(ser, command, expected_bytes, retries=3):
    for attempt in range(retries):
        try:
            ser.write(command)
            ser.flush()

            response = b''
            expected_length = 3 + expected_bytes + 2
            timeout = 3
            start_time = time.time()

            while (time.time() - start_time) < timeout:
                if ser.in_waiting:
                    new_data = ser.read(ser.in_waiting)
                    response += new_data
                    if len(response) >= expected_length:
                        break
                time.sleep(0.1)

            if len(response) != expected_length:
                raise Exception(f"读取失败: 预期{expected_length}字节，实际接收{len(response)}字节")

            received_crc = response[-2:]
            calculated_crc = crc16(response[:-2])
            if received_crc != calculated_crc:
                raise Exception("CRC校验失败")

            return response[3:-2]
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)

def interpret_value(data):
    value = struct.unpack('>i', data)[0]
    return value / 100

class SD76Device:
    def __init__(self, port='COM3', baudrate=9600):
        self.ser = serial.Serial(port, baudrate, bytesize=8, parity='N', stopbits=1, timeout=1)

    def read_upper_value(self):
        command = bytes.fromhex("01 03 00 21 00 02 94 01")
        upper_value_bytes = read_register(self.ser, command, 4)
        return interpret_value(upper_value_bytes)

    def read_lower_value(self):
        command = bytes.fromhex("01 03 00 23 00 02 35 C1")
        lower_value_bytes = read_register(self.ser, command, 4)
        return interpret_value(lower_value_bytes)

    def clear_upper_value(self):
        command = bytes.fromhex("01 06 00 00 00 03 C9 CB")
        self.ser.write(command)
        self.ser.flush()
        time.sleep(0.2)
        response = self.ser.read(8)
        return response == command[:8]

    def set_alarm_values(self, alarm1_value, alarm2_value):
        alarm1_hex = self.format_alarm_value(alarm1_value)
        alarm2_hex = self.format_alarm_value(alarm2_value)
        command_data = bytes.fromhex(f"01 10 00 0F 00 04 08 00 {alarm1_hex} 00 {alarm2_hex} ")
        crc = crc16(command_data)
        command = command_data + crc
        self.ser.write(command)
        response = self.ser.read(8)
        return len(response) == 8 and response[:6] == command[:6]


    def format_alarm_value(self,value):
        """将数值格式化为间隔2位带空格的6位字符串"""
        value_str = f"{value:06d}"
        value_out = f"{value_str[:2]} {value_str[2:4]} {value_str[4:]}"
        print("string",value_out)
        return value_out

    def close(self):
        if self.ser.is_open:
            self.ser.close()