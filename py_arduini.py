import tkinter as tk
from tkinter import messagebox
import serial.tools.list_ports
import time

class MotorControlApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Motor Control")
        
        self.Pos_ini = [[0, 90, 180] for _ in range(16)]  # 默认的 Pos1、Pos2 和 Pos3
        self.Pos_ini[0] =[10,0,95]
        self.Pos_ini[1] =[30,80,120]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        self.Pos_ini[2] =[10,80,140]
        self.Pos_ini[3] =[0,90,180]
        self.Pos_ini[4] =[0,90,180]
        self.Pos1_values = [tk.DoubleVar(value=Pos[0]) for Pos in self.Pos_ini]
        self.Pos2_values = [tk.DoubleVar(value=Pos[1]) for Pos in self.Pos_ini]
        self.Pos3_values = [tk.DoubleVar(value=Pos[2]) for Pos in self.Pos_ini]
        self.current_positions = [tk.DoubleVar() for _ in range(16)]
        
        self.serial_ports = self.get_serial_ports()
        self.selected_port = tk.StringVar()
        
        self.init_serial()
        self.create_widgets()

    def get_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def init_serial(self):
        if self.serial_ports:
            self.selected_port.set(self.serial_ports[0])
        else:
            messagebox.showerror("No Serial Ports", "No serial ports available.")
            self.master.destroy()
            return
        
        self.ser = serial.Serial(self.selected_port.get(), 9600, timeout=1)
        time.sleep(2)  # 等待串口初始化完成
    
    def create_widgets(self):
        port_label = tk.Label(self.master, text="Select Serial Port:")
        port_label.grid(row=0, column=0)
        
        port_menu = tk.OptionMenu(self.master, self.selected_port, *self.serial_ports)
        port_menu.grid(row=0, column=1)
        
        connect_button = tk.Button(self.master, text="Connect", command=self.connect_serial)
        connect_button.grid(row=0, column=2)
        
        for i in range(16):
            label = tk.Label(self.master, text=f"Motor {i+1}")
            label.grid(row=i+1, column=0)
            
            Pos1_entry = tk.Entry(self.master, textvariable=self.Pos1_values[i])
            Pos1_entry.grid(row=i+1, column=1)
            
            Pos2_entry = tk.Entry(self.master, textvariable=self.Pos2_values[i])
            Pos2_entry.grid(row=i+1, column=2)
            
            Pos3_entry = tk.Entry(self.master, textvariable=self.Pos3_values[i])
            Pos3_entry.grid(row=i+1, column=3)
            
            position_label = tk.Label(self.master, textvariable=self.current_positions[i])
            position_label.grid(row=i+1, column=4)
            
            move_to_Pos1_button = tk.Button(self.master, text="Pos-1", command=lambda idx=i: self.move_to_position(idx, 1))
            move_to_Pos1_button.grid(row=i+1, column=5)
            
            move_to_Pos2_button = tk.Button(self.master, text="Pos-2", command=lambda idx=i: self.move_to_position(idx, 2))
            move_to_Pos2_button.grid(row=i+1, column=6)

            move_to_Pos3_button = tk.Button(self.master, text="Pos-3", command=lambda idx=i: self.move_to_position(idx, 3))
            move_to_Pos3_button.grid(row=i+1, column=7)
        
        set_default_Pos1_button = tk.Button(self.master, text="Set Default Pos1", command=self.set_default_Pos1)
        set_default_Pos1_button.grid(row=18, column=0, columnspan=6)
    
    def connect_serial(self):
        if self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(self.selected_port.get(), 9600, timeout=1)
        time.sleep(2)  # 等待串口初始化完成
    
    def move_to_position(self, motor_index, pos_index):
        int_value = int(self.Pos1_values[motor_index].get()) if pos_index == 1 else int(self.Pos2_values[motor_index].get()) if pos_index == 2 else int(self.Pos3_values[motor_index].get())
        command = f"{motor_index},{int_value};"
        self.ser.write(command.encode())
        print(f"Sent command: {command}")
        time.sleep(0.1)
        response = self.ser.readline().decode().strip()
        print(f"Received response: {response}")
        self.current_positions[motor_index]=1.0
        #if response == "OK":
         #   messagebox.showinfo("Move Motor", f"Moved Motor {motor_index+1} to {'Pos1' if pos_index == 1 else 'Pos2' if pos_index == 2 else 'Pos3'} position")
        #else:
         #   messagebox.showerror("Move Motor", f"Failed to move Motor {motor_index+1} to {'Pos1' if pos_index == 1 else 'Pos2' if pos_index == 2 else 'Pos3'} position")
    
    def set_default_Pos1(self):
        for i in range(16):
            command = f"{i},{int(self.Pos1_values[i].get())}"
            self.ser.write(command.encode())
            print(f"Sent Default command: {command}")
            time.sleep(0.5)
            response = self.ser.readline().decode().strip()
            print(f"Received response: {response}")
            #if response != "OK":
            #    messagebox.showerror("Set Default Pos1", f"Failed to set Motor {i+1} to default Pos1 position")
            #else:
            #    self.current_positions[i].set(self.Pos1_values[i].get())  # 更新显示的当前位置为Pos1

def main():
    root = tk.Tk()
    app = MotorControlApp(root)
    root.mainloop()



main()