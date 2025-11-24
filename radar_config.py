# collect data from TI DCA1000 EVM

import serial
import time
import socket

# Radar EVM setting
class SerialConfig():
    def __init__(self, name, CLIPort, BaudRate):
        self.name = name
        self.CLIPort = serial.Serial(CLIPort, baudrate=BaudRate)

    def close(self):
        self.CLIPort.close()
        
    def SendConfig(self, ConfigFileName):
        lines = list(open(ConfigFileName))
        total_lines = len(lines)
        
        for idx, line in enumerate(lines):
            # 跳过空行和注释行
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('%'):
                continue
                
            self.CLIPort.write((line.rstrip('\r\n') + '\n').encode())
            print(f"Sent [{idx+1}/{total_lines}]: {stripped_line}")

            # 等待并读取返回数据（带超时，减少等待时间）
            start_time = time.time()
            timeout = 0.05  # 减少超时时间到50ms，加快配置速度
            response = b''
            
            while time.time() - start_time < timeout:
                if self.CLIPort.in_waiting > 0:
                    response += self.CLIPort.read(self.CLIPort.in_waiting)
                    # 如果已经收到响应，可以提前退出
                    if len(response) > 0:
                        break
                time.sleep(0.005)  # 减少sleep时间，提高响应速度

            if response:
                print(f"Received: {response.decode(errors='ignore').strip()}")
            
            
            time.sleep(0.005)

    def StartRadar(self):
        self.CLIPort.write('sensorStart\n'.encode())
        print('sensorStart\n')

    def StopRadar(self):
        self.CLIPort.write('sensorStop\n'.encode())
        print('sensorStop\n')

    def DisconnectRadar(self):
        self.CLIPort.write('sensorStop\n'.encode())
        self.CLIPort.close()

# DCA1000
class DCA1000Config():
    # config_address为控制参数地址，FPGA_address_cfg为数据参数地址
    def __init__(self, name, config_address, FPGA_address_cfg):
        self.name = name
        self.config_address = config_address
        self.FPGA_address_cfg = FPGA_address_cfg
        cmd_order = ['9', 'E', '3', 'B', '5', '6']
        self.sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sockConfig.bind(config_address)
        for k in range(5):
            # Send the command
            self.sockConfig.sendto(self.send_cmd(cmd_order[k]), FPGA_address_cfg)
            time.sleep(0.1)
            # Request data back on the config port
            msg, server = self.sockConfig.recvfrom(2048)
            # print('receive command:', msg.hex())

    def DCA1000_close(self):
        self.sockConfig.sendto(self.send_cmd('6'), self.FPGA_address_cfg)
        self.sockConfig.close()

    def send_cmd(self, code):
        # command code list
        CODE_1 = (0x01).to_bytes(2, byteorder='little', signed=False)
        CODE_2 = (0x02).to_bytes(2, byteorder='little', signed=False)
        CODE_3 = (0x03).to_bytes(2, byteorder='little', signed=False)
        CODE_4 = (0x04).to_bytes(2, byteorder='little', signed=False)
        CODE_5 = (0x05).to_bytes(2, byteorder='little', signed=False)
        CODE_6 = (0x06).to_bytes(2, byteorder='little', signed=False)
        CODE_7 = (0x07).to_bytes(2, byteorder='little', signed=False)
        CODE_8 = (0x08).to_bytes(2, byteorder='little', signed=False)
        CODE_9 = (0x09).to_bytes(2, byteorder='little', signed=False)
        CODE_A = (0x0A).to_bytes(2, byteorder='little', signed=False)
        CODE_B = (0x0B).to_bytes(2, byteorder='little', signed=False)
        CODE_C = (0x0C).to_bytes(2, byteorder='little', signed=False)
        CODE_D = (0x0D).to_bytes(2, byteorder='little', signed=False)
        CODE_E = (0x0E).to_bytes(2, byteorder='little', signed=False)

        # packet header & footer
        header = (0xA55A).to_bytes(2, byteorder='little', signed=False)
        footer = (0xEEAA).to_bytes(2, byteorder='little', signed=False)

        # data size
        dataSize_0 = (0x00).to_bytes(2, byteorder='little', signed=False)
        dataSize_6 = (0x06).to_bytes(2, byteorder='little', signed=False)

        # data
        data_FPGA_config = (0x01020102031e).to_bytes(6, byteorder='big', signed=False)
        data_packet_config = (0xc005350c0000).to_bytes(6, byteorder='big', signed=False)

        # connect to DCA1000
        connect_to_FPGA = header + CODE_9 + dataSize_0 + footer
        read_FPGA_version = header + CODE_E + dataSize_0 + footer
        config_FPGA = header + CODE_3 + dataSize_6 + data_FPGA_config + footer
        config_packet = header + CODE_B + dataSize_6 + data_packet_config + footer
        start_record = header + CODE_5 + dataSize_0 + footer
        stop_record = header + CODE_6 + dataSize_0 + footer

        if code == '9':
            re = connect_to_FPGA
        elif code == 'E':
            re = read_FPGA_version
        elif code == '3':
            re = config_FPGA
        elif code == 'B':
            re = config_packet
        elif code == '5':
            re = start_record
        elif code == '6':
            re = stop_record
        else:            
            re = 'NULL'
        # print('send command:', re.hex())
        return re