import numpy as np
from collections import deque

import dsp
from dsp.doppler_processing import doppler_processing # 多普勒FFT处理
import dsp.range_processing as range_processing # 距离FFT处理
import dsp.angle_estimation as Angle_dsp # AOA算法（Capon, MUSIC）
import dsp.utils as utils # 工具库（窗函数等）
import dsp.compensation as Compensation # 杂波抑制
from dsp.utils import Window # 工具库（窗函数等）
import globalvar as gl # 全局状态管理（如手势标志）


# 文件函数结构如下：
# doppler_fft()：多普勒维2D FFT封装（用于生成DTI图）
# RDA_Time()：时间-距离图、时间-多普勒图、距离-多普勒图的主函数
# Range_Angle()：生成距离-方位角图与距离-俯仰角图的主函数



#  图像帧缓存队列
rti_queue = deque(maxlen=12) #时间-距离图（1D Range）
rdi_queue = deque(maxlen=12) #距离-多普勒图（2D Range-Doppler）
rai_queue = deque(maxlen=12) #距离-方位角图
rei_queue = deque(maxlen=12) #距离-俯仰角图
# 关于deque
# deque是定双端队列，deque(maxlen=12)即创建一个最多容纳12个元素的双端队列，
# 双端队列（deque，全名double-ended queue）是一种具有队列和栈性质的抽象数据类型，
# 双端队列中的元素可以从两端弹出，插入和删除操作限定在队列的两边进行。
# deque比list在首部插入或删除时效率高得多（deque为o(1)时间，而list为o(n)时间）
# maxlen=12即双端队列最大长度为12，当队列内已经存在12个元素时，再向队尾添加新元素会导致队首元素被删除
# 这里使用队列是为了存储12帧的数据，方便做时间维的堆叠/平均，同时限制内存使用，始终只保留最新一段时间的雷达帧
# 是一种常见的滑动窗口缓存策略


gesturetimecnt = 0
# 用于计数手悬停事件

gesturetimecnt2 = 0

NUM_TX = 3
NUM_RX = 4
VIRT_ANT = 4
VIRT_ANT1 = 1
# Data specific parameters
NUM_CHIRPS = 64
NUM_ADC_SAMPLES = 64
# 2025-11-24 modify
# RANGE_RESOLUTION = .0488
RANGE_RESOLUTION = 0.09
# DOPPLER_RESOLUTION = 0.0806
DOPPLER_RESOLUTION = 0.14
NUM_FRAMES = 300

# DSP processing parameters
SKIP_SIZE = 4  # 忽略边缘角度的目标
ANGLE_RES = 0.5  # 角度分辨率
ANGLE_RANGE = 90  # 监督范围（扫描角度）
ANGLE_FFT_BINS= 64
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 64

numRangeBins = NUM_ADC_SAMPLES
numDopplerBins = NUM_CHIRPS

# 计算分辨率
range_resolution, bandwidth = dsp.range_resolution(NUM_ADC_SAMPLES)
doppler_resolution = dsp.doppler_resolution(bandwidth)

# Start DSP processing
range_azimuth = np.zeros((int(ANGLE_BINS), BINS_PROCESSED))
range_elevation = np.zeros((int(ANGLE_BINS), BINS_PROCESSED))
azimuth_elevation = np.zeros((ANGLE_FFT_BINS, ANGLE_FFT_BINS, NUM_ADC_SAMPLES))

# 导向矢量（用于AOA估计）
# 输出导向矢量矩阵，用于 Capon/MUSIC 算法扫描角度空间
num_vec, steering_vec = Angle_dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT) #theta跨度 theta分辨率 Vrx天线信号的数量


# doppler_fft()：
def doppler_fft(x,window_type_2d=None):
    fft2d_in = np.transpose(x, axes=(1, 0))  # rangbin chirps
    if window_type_2d: 
        fft2d_in = utils.windowing(fft2d_in, window_type_2d, axis=1)
    # 这里用zoom——FTT，是不是更加有效提取有效频率信息呢
    fft2d_out = np.fft.fft(fft2d_in, axis=1) # frame rangbin dopplerbin
    fft2d_log_abs = np.log2(np.abs(fft2d_out))
    # numADCSamples, numChirpsPerFrame
    det_matrix_vis = np.fft.fftshift(fft2d_log_abs, axes=1)
    return det_matrix_vis



# 该函数主要是画三个图 
# RTI 时间距离图（沿chirps-samples矩阵的纵轴做1D-FFT得到的时间距离图）
# DTI 时间多普勒图（时频图）
# ATI 时间方位角图
framecnt  = 0
def RDA_Time(adc_data, window_type_1d=None, clutter_removal_enabled=True, CFAR_enable=False, axis=-1):

    global gesturetimecnt, framecnt
    #  转换成(num_chirps_per_frame, num_rx_antennas, num_adc_samples)
    adc_data = np.transpose(adc_data, [0, 2, 1])
    # 不做下采样
    radar_cube = range_processing(2*adc_data[:,:,0:64:1], window_type_1d, 2)

    if clutter_removal_enabled:
        radar_cube = Compensation.clutter_removal(radar_cube,axis=0)


    # 距离多普勒图
    range_doppler_fft, aoa_input = doppler_processing(radar_cube, 
                                            num_tx_antennas=3, 
                                            interleaved=False, 
                                            clutter_removal_enabled=False, #前面已经做了
                                            window_type_2d=Window.HANNING,
                                            accumulate = False)

    rdi_abs = np.transpose(np.fft.fftshift(np.abs(range_doppler_fft), axes=2), [0, 2, 1])
    rdi_abs = np.flip(rdi_abs, axis=0)
    rdi_queue.append(rdi_abs)
    # 16个frame叠加返回
    rdi_framearray = np.array(rdi_queue)#frame chirps adcnum numVirtualAntennas

    # 时间距离图
    det_matrix = radar_cube[:, 0, :]

    #用距离图作为判断 
    # [4:36,]是指4~36这个rangbin区间，有true的隔宿大于26个
    Iscapture = gl.get_value('IsRecognizeorCapture')
    if(np.sum(det_matrix[:,36:62]>3e3)>14) :
        if Iscapture:
            gesturetimecnt = gesturetimecnt + 1

    if(gesturetimecnt>=2) and Iscapture:
        framecnt = framecnt + 1
        # framecnt是用来相当延迟多少帧再截图到judgegesture文件夹中
        if framecnt>=8:
            if gl.get_value('timer_2s'):
                gl.set_value('usr_gesture',True)    
                gl.set_value('timer_2s',False)
            framecnt = 0
            gesturetimecnt=0
            

    rti_queue.append(det_matrix)
    rti_framearray = np.array(rti_queue)#frame chirps adcnum
    rti_array = np.reshape(rti_framearray, (1, -1, 64))#chirps adcnum
    # (num_chirps_per_frame, num_range_bins, num_rx_antennas)
    rti_array_out = np.transpose(rti_array, [1, 2, 0])

    # 微多普勒时间图（时频图）
    micro_doppler_data = np.zeros((rti_framearray.shape[0], rti_framearray.shape[1], rti_framearray.shape[2]), dtype=np.float64)
    micro_doppler_data_out = np.zeros((16,64), dtype=np.float64)
    for i, frame in enumerate(rti_framearray):
            # --- Show output
            det_matrix_vis = doppler_fft(frame,window_type_2d=Window.HANNING)
            micro_doppler_data[i,:,:] = det_matrix_vis

    
    rti_array_out = np.flip(np.abs(rti_array_out), axis=1)
    # 
    rti_array_out[rti_array_out<3e3]=0
    # # 用RDI图判断信噪比强度64*64
    # if(np.sum(rti_array_out[0:1024:16,:,:]<100)>4090):
    #     SNR = False
    # else:
    #     SNR = True

    micro_doppler_data_out = micro_doppler_data.sum(axis=1)
    micro_doppler_data_out[micro_doppler_data_out<20]=0

    return rti_array_out, rdi_framearray, micro_doppler_data_out



def Range_Angle(data,  padding_size=None, clutter_removal_enabled=True, window_type_1d = Window.HANNING,Music_enable = False):
    # (0:TX1-RX1,1:TX1-RX2,2:TX1-RX3,3:TX1-RX4,| 4:TX2-RX1,5:TX2-RX2,6:TX2-RX3,7:TX2-RX4,| 8:TX3-RX1,9:TX3-RX2,10:TX3-RX3,11:TX3-RX4)
    # data = np.fft.fft2(data[:, :, [1,0,9,8]], s=[padding_size[0], padding_size[1]], axes=[0, 1])
    # global SNR

    #  转换成(num_chirps_per_frame, num_rx_antennas, num_adc_samples)
    adc_data = np.transpose(data, [0, 2, 1])
    # radar_cube = dsp.zoom_range_processing(adc_data, 0.1, 0.5, 1, 0, adc_data.shape[2])
    # 不做下采样
    radar_cube = range_processing(2*adc_data[:,:,0:64:1], window_type_1d, 2)

    if clutter_removal_enabled:
        radar_cube = Compensation.clutter_removal(radar_cube,axis=0)

    # np.save('data.npy',radar_cube)
    frame_SNR = np.log(np.sum(np.abs(radar_cube[:,:])))-14.7
    # 改进：使用更温和的SNR阈值，避免完全清零数据
    # 如果SNR太低，使用最小值而不是0，保留部分信息
    if(np.abs(frame_SNR)<1.8):
        frame_SNR = 0.8  # 改为最小值而不是0，避免完全丢失角度信息
    # print(frame_SNR)
    # --- capon beamforming
    beamWeights   = np.zeros((VIRT_ANT, BINS_PROCESSED), dtype=np.complex_)

    # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
    # has doppler at the last dimension.
    # 方位角
    for i in range(BINS_PROCESSED):
        if Music_enable:
            range_azimuth[:,i] = dsp.aoa_music_1D(steering_vec, radar_cube[:, [10,8,6,4], i].T, num_sources=1)
        else:                                                                  #4,6,8,10#
            range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, [7,4,3,0], i].T, steering_vec, magnitude=True)
            # range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon_new(radar_cube[:, [10,8,6,4], i].T,radar_cube[:, [10,8,6,4], i+1].T, steering_vec, magnitude=True)
    # 俯仰角
    for i in range(BINS_PROCESSED): 
        if Music_enable:
            range_elevation[:,i] = dsp.aoa_music_1D(steering_vec, radar_cube[:, [1,0,9,8], i].T, num_sources=1)
        else:
            # radar_cube[:, [1,0,9,8], i].T*[[1],[-1],[1],[-1]] 不用这个，和导向矢量有关系把
            # 1 -1 1 -1为俯仰角补偿 方位角涉及到的四个虚拟阵元都是同相位的 所以不需要
            range_elevation[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, [7,6,11,10], i].T*[[1],[-1],[1],[-1]], steering_vec, magnitude=True)

    rdi_ab1 = np.flip(np.abs(range_azimuth), axis=1)
    rdi_ab2 = np.flip(np.abs(range_elevation), axis=1)
    # 改进：移除限幅操作，保留完整的峰值信息以提高角度精度
    # rdi_ab1 = np.minimum(rdi_ab1,rdi_ab1.max()/2)
    # rdi_ab2 = np.minimum(rdi_ab2,rdi_ab2.max()/2)
    # 把不在手势范围的目标去除

    # 修改：去除手势范围外的区域会导致距离方位角和距离俯仰角图显示不全
    # rdi_ab1[:,40:90] = 0
    # rdi_ab2[:,40:90] = 0


    # rdi_ab1[:,13:19] = 0.1*rdi_ab1[:,13:19] 
    # rdi_ab2[:,13:19] = 0.1*rdi_ab2[:,13:19] 
    # rdi_ab1[:5,:] = 0
    # rdi_ab2[-5:,:] = 0
    # 加权 信噪比（改进：避免除零错误，使用更稳定的归一化）
    if rdi_ab1.max() > 0:
        rdi_ab1 = rdi_ab1 / rdi_ab1.max() * frame_SNR
    else:
        rdi_ab1 = rdi_ab1 * 0  # 如果全为0，保持为0
    if rdi_ab2.max() > 0:
        rdi_ab2 = rdi_ab2 / rdi_ab2.max() * frame_SNR
    else:
        rdi_ab2 = rdi_ab2 * 0  # 如果全为0，保持为0
    
    # 改进：添加轻微的角度维平滑，减少噪声影响（可选）
    # 使用简单的移动平均进行轻微平滑
    try:
        # 对角度维进行轻微平滑（保持距离维不变），使用3点移动平均
        kernel = np.array([0.25, 0.5, 0.25])  # 简单的平滑核
        # 手动实现1D卷积
        smoothed_ab1 = np.zeros_like(rdi_ab1)
        smoothed_ab2 = np.zeros_like(rdi_ab2)
        for i in range(rdi_ab1.shape[0]):
            if i == 0:
                smoothed_ab1[i] = rdi_ab1[i] * 0.5 + rdi_ab1[min(i+1, rdi_ab1.shape[0]-1)] * 0.5
                smoothed_ab2[i] = rdi_ab2[i] * 0.5 + rdi_ab2[min(i+1, rdi_ab2.shape[0]-1)] * 0.5
            elif i == rdi_ab1.shape[0] - 1:
                smoothed_ab1[i] = rdi_ab1[i-1] * 0.5 + rdi_ab1[i] * 0.5
                smoothed_ab2[i] = rdi_ab2[i-1] * 0.5 + rdi_ab2[i] * 0.5
            else:
                smoothed_ab1[i] = rdi_ab1[i-1] * 0.25 + rdi_ab1[i] * 0.5 + rdi_ab1[i+1] * 0.25
                smoothed_ab2[i] = rdi_ab2[i-1] * 0.25 + rdi_ab2[i] * 0.5 + rdi_ab2[i+1] * 0.25
        rdi_ab1 = smoothed_ab1
        rdi_ab2 = smoothed_ab2
    except:
        # 如果平滑失败，使用原始数据
        pass

    rai_queue.append(rdi_ab1)
    rei_queue.append(rdi_ab2)
    # 16个frame叠加返回
    rai_framearray = np.array(rai_queue)#frame chirps adcnum 
    rei_framearray = np.array(rei_queue)#frame chirps adcnum 

    return rai_framearray, rei_framearray


def extract_pointcloud_from_angle_maps(rai_data, rei_data, range_resolution=RANGE_RESOLUTION, 
                                       angle_resolution=ANGLE_RES, angle_range=ANGLE_RANGE,
                                       threshold_ratio=0.3):
    """
    基于网格检测的点云提取算法
    根据雷达参数（距离分辨率、角度分辨率）将空间划分成网格，
    检测每个网格内是否存在超过阈值的目标，并输出为3D点云

    Args:
        rai_data: 距离-方位角图，形状 (angle_bins, range_bins) 或 (frame, angle_bins, range_bins)
        rei_data: 距离-俯仰角图，形状 (angle_bins, range_bins) 或 (frame, angle_bins, range_bins)
        range_resolution: 距离分辨率（米）
        angle_resolution: 角度分辨率（度）
        angle_range: 角度范围（度），从-angle_range到+angle_range
        threshold_ratio: 检测阈值比例（相对于最大值）
    
    Returns:
        pointcloud: numpy数组，形状 (num_points, 4)，每行为 [range, x, y, z]
    """
    try:
        # 如果输入是多帧数据，取最新的一帧
        if len(rai_data.shape) == 3:
            rai = rai_data[-1]  # 取最新一帧
            rei = rei_data[-1]
        else:
            rai = rai_data
            rei = rei_data
        
        # 确保数据是2D的
        if len(rai.shape) != 2 or len(rei.shape) != 2:
            return np.zeros((0, 4), dtype=np.float32)
        
        azimuth_bins, range_bins = rai.shape
        elevation_bins, _ = rei.shape
        
        # 计算角度索引到实际角度的映射
        # 方位角：从-angle_range到+angle_range
        azimuth_indices = np.arange(azimuth_bins)
        azimuth_angles_deg = -angle_range + azimuth_indices * angle_resolution
        
        # 俯仰角：从-angle_range到+angle_range
        elevation_indices = np.arange(elevation_bins)
        elevation_angles_deg = -angle_range + elevation_indices * angle_resolution
        
        # 计算距离索引到实际距离的映射
        range_indices = np.arange(range_bins)
        ranges = range_indices * range_resolution
        
        # 计算阈值
        rai_max = np.max(rai)
        rei_max = np.max(rei)
        
        # 如果数据全为0或最大值太小，返回空点云
        if rai_max <= 0 or rei_max <= 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        # 计算阈值（使用相对阈值）
        rai_threshold = rai_max * threshold_ratio
        rei_threshold = rei_max * threshold_ratio
        
        # 确保阈值不会太低（至少5%）
        rai_threshold = max(rai_threshold, rai_max * 0.05)
        rei_threshold = max(rei_threshold, rei_max * 0.05)
        
        # 网格检测：遍历所有(距离, 方位角, 俯仰角)网格
        pointcloud_list = []
        
        # 遍历所有距离bin
        for range_idx in range(range_bins):
            range_val = ranges[range_idx]
            
            # 过滤掉距离为0或过近的点（可能是噪声）
            if range_val < 0.1:  # 小于10cm的点认为是噪声
                continue
            
            # 获取该距离bin的方位角和俯仰角数据
            azimuth_profile = rai[:, range_idx]  # 形状: (azimuth_bins,)
            elevation_profile = rei[:, range_idx]  # 形状: (elevation_bins,)
            
            # 遍历所有方位角bin
            for azimuth_idx in range(azimuth_bins):
                azimuth_val = azimuth_profile[azimuth_idx]
                
                # 如果方位角值超过阈值，检查对应的俯仰角
                if azimuth_val > rai_threshold:
                    # 遍历所有俯仰角bin，找到最强的俯仰角
                    best_elevation_idx = np.argmax(elevation_profile)
                    elevation_val = elevation_profile[best_elevation_idx]
                    
                    # 如果俯仰角值也超过阈值，或者方位角值足够强，则生成点
                    if elevation_val > rei_threshold or azimuth_val > rai_threshold * 1.5:
                        # 获取实际角度
                        azimuth_deg = azimuth_angles_deg[azimuth_idx]
                        elevation_deg = elevation_angles_deg[best_elevation_idx]
                        
                        # 转换为弧度
                        azimuth_rad = np.deg2rad(azimuth_deg)
                        elevation_rad = np.deg2rad(elevation_deg)
                        
                        # 球坐标转笛卡尔坐标
                        # 雷达坐标系（IWR6843横向放置，面朝目标）：
                        # - X轴：前方（雷达正对方向）
                        # - Y轴：左侧（方位向，已反转）
                        # - Z轴：上方（俯仰向）
                        # 方位角：从X轴正方向逆时针旋转（左侧为正，右侧为负）
                        # 俯仰角：从XY平面向上为正，向下为负
                        x = range_val * np.cos(elevation_rad) * np.cos(azimuth_rad)
                        y = -range_val * np.cos(elevation_rad) * np.sin(azimuth_rad)  # Y轴反转
                        z = range_val * np.sin(elevation_rad)
                        
                        # 存储为四元组 [range, x, y, z]
                        pointcloud_list.append([range_val, x, y, z])
            
            # 也检查俯仰角图中超过阈值的点（可能方位角较弱但俯仰角强）
            for elevation_idx in range(elevation_bins):
                elevation_val = elevation_profile[elevation_idx]
                
                if elevation_val > rei_threshold:
                    # 找到最强的方位角
                    best_azimuth_idx = np.argmax(azimuth_profile)
                    azimuth_val = azimuth_profile[best_azimuth_idx]
                    
                    # 如果方位角值也超过阈值，或者俯仰角值足够强，则生成点
                    if azimuth_val > rai_threshold or elevation_val > rei_threshold * 1.5:
                        # 检查是否已经添加过这个点（避免重复）
                        azimuth_deg = azimuth_angles_deg[best_azimuth_idx]
                        elevation_deg = elevation_angles_deg[elevation_idx]
                        
                        # 转换为弧度
                        azimuth_rad = np.deg2rad(azimuth_deg)
                        elevation_rad = np.deg2rad(elevation_deg)
                        
                        # 球坐标转笛卡尔坐标
                        x = range_val * np.cos(elevation_rad) * np.cos(azimuth_rad)
                        y = -range_val * np.cos(elevation_rad) * np.sin(azimuth_rad)  # Y轴反转
                        z = range_val * np.sin(elevation_rad)
                        
                        # 检查是否已存在相似的点（避免重复添加）
                        is_duplicate = False
                        for existing_point in pointcloud_list:
                            if abs(existing_point[1] - x) < 0.05 and \
                               abs(existing_point[2] - y) < 0.05 and \
                               abs(existing_point[3] - z) < 0.05:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            pointcloud_list.append([range_val, x, y, z])
        
        if len(pointcloud_list) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        return np.array(pointcloud_list, dtype=np.float32)
    except Exception as e:
        print(f"点云提取函数错误: {e}, RAI形状: {rai_data.shape if hasattr(rai_data, 'shape') else 'N/A'}, REI形状: {rei_data.shape if hasattr(rei_data, 'shape') else 'N/A'}")
        import traceback
        traceback.print_exc()
        return np.zeros((0, 4), dtype=np.float32)

  