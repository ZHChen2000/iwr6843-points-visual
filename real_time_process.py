import threading as th # 多线程模块
import numpy as np # 用于定义大容量数据缓冲区及矩阵操作
import DSP # 自定义雷达信号处理模块
from dsp.utils import Window # 信号窗函数配置（如HANNING窗）
from ctypes import *
import globalvar as gl # 全局变量管理模块
# ctypes用于将numpy缓冲区转换为C语言指针
# *为通配符，意思是将ctypes中的所有成员引入到该文件中

# 参数配置 Parameters
adc_sample = 64
chirp = 64
tx_num = 3
rx_num = 4
IQ_channel = 2
frame_length = adc_sample * chirp * tx_num * rx_num * IQ_channel

# UDPCAPTUREADCRAWDATA.dll是C语言写的动态链接库，可实现高性能UDP包捕获
# .dll = .Dynamic-Link Library
dll = cdll.LoadLibrary('libs/UDPCAPTUREADCRAWDATA.dll')

# a是双缓冲标志位
a = np.zeros(1).astype(int)

# b是原始数据缓冲区，长度为两个完整雷达数据帧，frame_length*2就是缓冲2个数据帧，格式为 c_short
# c_short长度为16位，用于表示复数数据
b = np.zeros(frame_length*2).astype(c_short)

# 转换为ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用
# 将 NumPy数组转换为C指针，供DLL读取和写入
a_ctypes_ptr = cast(a.ctypes.data, POINTER(c_int))
# 转换为 ctypes，这里转换后的可以直接利用ctypes转换为c语言中的int*，然后在c中使用，供DLL读取和写入
b_ctypes_ptr = cast(b.ctypes.data, POINTER(c_short))


# 原始数据捕获线程，启动线程后，将开始执行run()函数
class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length):
        th.Thread.__init__(self, name=name)
        self.bin_data = bin_data # 成员变量：原始数据
        self.frame_length = data_frame_length # 成员变量：数据帧长度
    def run(self):
        global a_ctypes_ptr, b_ctypes_ptr
        # a是双缓冲标志位指针
        # b是缓冲区指针
        # 该线程是一个是一个后台独立线程，它将不断调用C函数captureudp()进行UDP数据采集
        # 接收到的数据会被写入b缓冲区中，同时通过a[0]指示哪个缓冲区已填满（a[0]和a[1]分别是两个缓冲区的标志位）
        dll.captureudp(a_ctypes_ptr, b_ctypes_ptr, self.frame_length)
        # run()函数阻塞调用C函数，由其内部循环控制
        # 该线程是阻塞线程，也就是说，会完全等到UDP采集完数据后再向后执行，因为DLL内部本质上会一直
        # 抓取来自端口的数据，所以自然不是非阻塞的线程


# 雷达信号处理线程
class DataProcessor(th.Thread):
    def __init__(self, name, config, bin_queue, rti_queue, dti_queue,  rdi_queue, rai_queue, rei_queue, pointcloud_queue=None):
        # config: 包含[adc_sample, chirp_num, tx_num, rx_num]等信息
        # bin_queue: 来自UDP缓冲区的原始数据队列
        # rti_queue、rei_queue等: 图像数据输出队列
        # pointcloud_queue: 点云数据输出队列
        th.Thread.__init__(self, name=name)
        self.adc_sample = config[0]
        self.chirp_num = config[1]
        self.tx_num = config[2]
        self.rx_num = config[3]
        self.bin_queue = bin_queue
        self.rti_queue = rti_queue
        self.dti_queue = dti_queue
        self.rdi_queue = rdi_queue
        self.rai_queue = rai_queue
        self.rei_queue = rei_queue
        self.pointcloud_queue = pointcloud_queue

    # 数据处理线程内部需要执行的内容
    def run(self):
        global frame_count
        frame_count = 0
        lastflar = 0
        while True:
            # 对应dll中的双缓冲区，0区和1区
            # 这个if判断用来监测UDP双缓冲区状态
            # 每当缓冲区标志位a[0]发生变化，说明新数据写入了某一块缓冲区（双缓冲机制）
            if(lastflar != a_ctypes_ptr[0]):
                lastflar = a_ctypes_ptr[0]
                # 读取对应缓冲区数据
                data = np.array(b_ctypes_ptr[frame_length*(1-a_ctypes_ptr[0]):frame_length*(2-a_ctypes_ptr[0])])

                # 解码复数IQ数据，每个样本由4个short组成：实部1、实部2、虚部1、虚部2，重组为复数对（I + jQ）
                data = np.reshape(data, [-1, 4])
                data = data[:, 0:2:] + 1j * data[:, 2::]
                # [num_chirps*tx_num, wuli_antennas, num_samples]

                # 三维矩阵重组（雷达帧）
                data = np.reshape(data, [self.chirp_num * self.tx_num, -1, self.adc_sample])
                # [num_chirps*tx_num, num_samples, wuli_antennas]

                data = data.transpose([0, 2, 1])
                # 192 = 64*3 记得改
                # TX1的：[num_chirps, num_samples, wuli_antennas]
                ch1_data = data[0: self.adc_sample*3: 3, :, :]
                # TX2的：[num_chirps, num_samples, wuli_antennas]
                ch2_data = data[1: self.adc_sample*3: 3, :, :]
                # TX3的：[num_chirps, num_samples, wuli_antennas]
                ch3_data = data[2: self.adc_sample*3: 3, :, :]
                # channel的排序方式：(0:TX1-RX1,1:TX1-RX2,2:TX1-RX3,3:TX1-RX4,| 4:TX2-RX1,5:TX2-RX2,6:TX2-RX3,7:TX2-RX4,| 8:TX3-RX1,9:TX3-RX2,10:TX3-RX3,11:TX3-RX4)
                data = np.concatenate([ch1_data, ch2_data, ch3_data], axis=2)

                frame_count += 1

                # 调用DSP库，计算时间-距离图、距离-多普勒图、时间-多普勒图
                rti, rdi, dti = DSP.RDA_Time(data, window_type_1d=Window.HANNING, axis=1)

                # 调用DSP库，计算距离-方位角图、距离俯仰角图
                rai, rei = DSP.Range_Angle(data, padding_size=[128, 64, 64])

                # 将解码后的图像数据送入队列供后续UI显示
                self.rti_queue.put(rti)
                self.dti_queue.put(dti)
                self.rdi_queue.put(rdi)
                self.rai_queue.put(rai)
                self.rei_queue.put(rei)
                
                # 从距离-方位角图和距离-俯仰角图中提取点云
                if self.pointcloud_queue is not None:
                    try:
                        # 从全局变量获取阈值参数
                        threshold_ratio = gl.get_value('pointcloud_threshold', 0.1)  # 默认值改为0.1，更宽松
                        pointcloud = DSP.extract_pointcloud_from_angle_maps(rai, rei, threshold_ratio=threshold_ratio)
                        
                        # 调试信息：每50帧打印一次
                        if frame_count % 50 == 0:
                            rai_max = np.max(rai) if rai.size > 0 else 0
                            rei_max = np.max(rei) if rei.size > 0 else 0
                            rai_shape = rai.shape if hasattr(rai, 'shape') else 'N/A'
                            rei_shape = rei.shape if hasattr(rei, 'shape') else 'N/A'
                            point_count = len(pointcloud) if pointcloud is not None and len(pointcloud) > 0 else 0
                            print(f"[点云提取] 帧{frame_count}: RAI形状={rai_shape}, 最大值={rai_max:.2f} | REI形状={rei_shape}, 最大值={rei_max:.2f} | 阈值={threshold_ratio:.2f} | 提取点数={point_count}")
                        
                        if pointcloud is not None and len(pointcloud) > 0:
                            # 如果队列满，移除最旧的数据
                            if self.pointcloud_queue.full():
                                try:
                                    self.pointcloud_queue.get_nowait()
                                except:
                                    pass
                            self.pointcloud_queue.put(pointcloud)
                    except Exception as e:
                        # 点云提取失败不影响主流程，但打印错误信息
                        if frame_count % 50 == 0:
                            import traceback
                            print(f"[点云提取错误] 帧{frame_count}: {e}")
                            traceback.print_exc()
                        pass
