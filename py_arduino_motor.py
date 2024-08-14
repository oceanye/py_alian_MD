import tkinter as tk
from tkinter import messagebox, ttk
import serial.tools.list_ports
import time


class MotorControl:
    def __init__(self):
        self.ser = None
        self.current_positions = [0 for _ in range(16)]
        self.Pos1_values = [0 for _ in range(16)]
        self.Pos2_values = [90 for _ in range(16)]
        self.Pos3_values = [180 for _ in range(16)]

    def init_serial(self, port):
        try:
            self.ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # 等待串口初始化完成
            print(f"Successfully opened port {port}")
        except serial.SerialException as e:
            messagebox.showerror("Serial Port Error", f"Could not open port {port}: {str(e)}")
            raise

    def motor_command(self, motor_index, value):
        if not self.ser or not self.ser.is_open:
            messagebox.showerror("Serial Port Error", "Serial port is not open.")
            return False

        command = f"{motor_index},{int(value)};"
        try:
            self.ser.write(command.encode())
            print(f"Sent command: {command}")
            time.sleep(0.1)  # 给设备一些响应时间
            response = self.ser.readline().decode().strip()
            print(f"Received response: {response}")

            if response == "OK":
                self.current_positions[motor_index] = value
                return True
            else:
                print(f"Unexpected response: {response}")
                return False
        except Exception as e:
            print(f"Error sending command: {e}")
            return False

    def move_to_position(self, motor_index, value):
        success = self.motor_command(motor_index, value)
        if success:
            print(f"Moved Motor {motor_index} to position {value}")
        else:
            print(f"Failed to move Motor {motor_index} to position {value}")
        return success


class MotorControlApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Motor Control")
        self.motor_control = MotorControl()
        self.current_positions_vars = [tk.DoubleVar(value=pos) for pos in self.motor_control.current_positions]
        self.Pos1_vars = [tk.DoubleVar(value=pos) for pos in self.motor_control.Pos1_values]
        self.Pos2_vars = [tk.DoubleVar(value=pos) for pos in self.motor_control.Pos2_values]
        self.Pos3_vars = [tk.DoubleVar(value=pos) for pos in self.motor_control.Pos3_values]
        self.create_widgets()

    def create_widgets(self):
        # 添加串口选择下拉框
        self.port_var = tk.StringVar()
        ttk.Label(self.master, text="Select Port:").grid(row=0, column=0)
        self.port_combo = ttk.Combobox(self.master, textvariable=self.port_var)
        self.port_combo.grid(row=0, column=1)
        self.refresh_button = ttk.Button(self.master, text="Refresh", command=self.refresh_ports)
        self.refresh_button.grid(row=0, column=2)
        self.connect_button = ttk.Button(self.master, text="Connect", command=self.connect_serial)
        self.connect_button.grid(row=0, column=3)

        self.refresh_ports()  # 初始化端口列表

        for i in range(16):
            label = tk.Label(self.master, text=f"Motor {i + 1}")
            label.grid(row=i + 2, column=0)

            Pos1_entry = tk.Entry(self.master, textvariable=self.Pos1_vars[i])
            Pos1_entry.grid(row=i + 2, column=1)

            Pos2_entry = tk.Entry(self.master, textvariable=self.Pos2_vars[i])
            Pos2_entry.grid(row=i + 2, column=2)

            Pos3_entry = tk.Entry(self.master, textvariable=self.Pos3_vars[i])
            Pos3_entry.grid(row=i + 2, column=3)

            position_label = tk.Label(self.master, textvariable=self.current_positions_vars[i])
            position_label.grid(row=i + 2, column=4)

            move_to_Pos1_button = tk.Button(self.master, text="Pos-1",
                                            command=lambda idx=i: self.move_to_position(idx, 1))
            move_to_Pos1_button.grid(row=i + 2, column=5)

            move_to_Pos2_button = tk.Button(self.master, text="Pos-2",
                                            command=lambda idx=i: self.move_to_position(idx, 2))
            move_to_Pos2_button.grid(row=i + 2, column=6)

            move_to_Pos3_button = tk.Button(self.master, text="Pos-3",
                                            command=lambda idx=i: self.move_to_position(idx, 3))
            move_to_Pos3_button.grid(row=i + 2, column=7)

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])

    def connect_serial(self):
        port = self.port_var.get()
        try:
            self.motor_control.init_serial(port)
            messagebox.showinfo("Connection", f"Successfully connected to {port}")
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))

    def move_to_position(self, motor_index, pos_index):
        if not self.motor_control.ser or not self.motor_control.ser.is_open:
            messagebox.showerror("Serial Port Error", "Please connect to a serial port first.")
            return

        if pos_index == 1:
            value = self.Pos1_vars[motor_index].get()
        elif pos_index == 2:
            value = self.Pos2_vars[motor_index].get()
        else:
            value = self.Pos3_vars[motor_index].get()

        success = self.motor_control.move_to_position(motor_index, value)

        #if success:
            #self.current_positions_vars[motor_index].set(value)
            #messagebox.showinfo("Move Motor", f"Moved Motor {motor_index + 1} to Pos{pos_index} position")
        #else:
           # messagebox.showerror("Move Motor", f"Failed to move Motor {motor_index + 1} to Pos{pos_index} position")


def run_gui():
    root = tk.Tk()
    app = MotorControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()