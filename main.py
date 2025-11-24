# ========== 必须在所有导入之前设置环境变量 ==========
import os
import multiprocessing
# 修复 Windows 上 joblib/loky 检测物理 CPU 核心数失败的问题
# 设置 LOKY_MAX_CPU_COUNT 为逻辑核心数，避免尝试检测物理核心数
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
# 注意：Windows 上 multiprocessing 不支持 'threading' 上下文，已移除 JOBLIB_START_METHOD 设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ====================================================

from real_time_process import UdpListener, DataProcessor
from radar_config import SerialConfig
from radar_config import DCA1000Config
from queue import Queue
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem, GLGridItem, GLAxisItem
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import time
import torch
import sys
import numpy as np
from serial.tools import list_ports
import iwr6843_tlv.detected_points as readpoint
import globalvar as gl
import pointcloud_clustering

# 在导入 pointcloud_clustering 后，尝试 patch joblib/loky（如果 pointcloud_clustering 已经导入了 sklearn）
# 这样可以确保即使 pointcloud_clustering 中导入了 sklearn，我们也能 patch
try:
    import joblib.externals.loky.backend.context as loky_context
    if hasattr(loky_context, '_count_physical_cores'):
        def _patched_count_physical_cores():
            """Patch 后的函数，直接返回逻辑核心数，避免调用 wmic"""
            return multiprocessing.cpu_count()
        loky_context._count_physical_cores = _patched_count_physical_cores
except (ImportError, AttributeError):
    # 如果模块不存在或没有该函数，忽略
    pass

import warnings
# 过滤 NumPy 版本警告（scikit-learn 发出的）
warnings.filterwarnings('ignore', category=UserWarning, message='.*NumPy version.*')
warnings.filterwarnings('ignore', message='.*A NumPy version.*')
# 过滤 joblib/loky 警告（更全面的过滤）
warnings.filterwarnings('ignore', message='.*LOKY_MAX_CPU_COUNT.*')
warnings.filterwarnings('ignore', message='.*Returning the number of logical cores.*')
warnings.filterwarnings('ignore', message='.*系统找不到指定的文件.*')
warnings.filterwarnings('ignore', message='.*WinError 2.*')
warnings.filterwarnings('ignore', message='.*_count_physical_cores.*')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')
# 过滤所有 UserWarning 中与 CPU 核心数相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib.*')
warnings.filterwarnings('ignore', category=UserWarning, module='loky.*')

# import models.predict as predict
# from models.model import CNet, FeatureFusionNet
import matplotlib.pyplot as plt
from colortrans import pg_get_cmap
import threading

# -----------------------------------------------
from UI_interface import Ui_MainWindow, Qt_pet
# -----------------------------------------------


# comments by ZHChen 2025-05-20

datasetfile = 'dataset'
datasetsencefile = ' '
gesturedict = {
                '0':'backward',
                '1':'dbclick',
                '2':'down',
                '3':'front',
                '4':'Left',
                '5':'Right',
                '6':'up',
                '7':'NO'
               }

cnt = 0

_flagdisplay = False

# 全局变量初始化
CLIport_name = ''
Dataport_name = ''

# 加载手势识别模型
def loadmodel():
    global model
    if (modelfile.currentText()!='--select--'and modelfile.currentText()!=''):
        model_info = torch.load(modelfile.currentText(),map_location='cpu')
        # TODO: 
        model = []
        model.load_state_dict(model_info['state_dict'])
        printlog('加载'+modelfile.currentText()+'模型成功!',fontcolor='blue')
    else:
        printlog("请加载模型!",fontcolor='red')

#
def cleartjpg():
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/"+str(7)+".jpg"))
    subWin.img_update("gesture_icons/"+str(7)+".jpg")   

# 提取多种特征并调用深度学习模型（预留位置）进行手势识别
def Judge_gesture(a,b,c,d,e):
    global _flagdisplay
    if model:
        # TODO:
        fanhui = [] #predict.predictGesture(model,d,b,e,c,a)
        view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/"+str(fanhui)+".jpg"))
        subWin.img_update("gesture_icons/"+str(fanhui)+".jpg")
        QtCore.QTimer.singleShot(2000, cleartjpg)
        _flagdisplay = True
        printlog("输出:" + gesturedict[str(fanhui)],fontcolor='blue')
        return gesturedict[str(fanhui)]

def get_new_strategy_params():
    """根据选择的灵敏度等级返回相应的参数"""
    global fall_new_strategy_sensitivity
    # 如果控件还未初始化，返回默认参数（中等灵敏度）
    if 'fall_new_strategy_sensitivity' not in globals() or fall_new_strategy_sensitivity is None:
        return {
            'time_window_ms': 3000,  # 时间窗口：3秒
            'height_threshold': 0.5,  # 高度下降阈值：0.5米
            'velocity_threshold': 0.5,  # 速度阈值：0.5 m/s
            'acceleration_threshold': 2.0,  # 加速度阈值：2.0 m/s²
            'low_height_threshold': 0.3,  # 低姿态高度阈值：0.3m
            'min_height_low': 0.5,  # 最低点阈值：0.5m
            'low_duration_threshold': 0.5,  # 低姿态持续时间阈值：0.5秒
            'min_conditions': 6,  # 最少满足条件数：6个（75%）
            'max_height_min': -2.0,  # 最大高度下限
            'max_height_max': 1.5,  # 最大高度上限
            'slow_velocity_threshold': 0.3  # 缓慢动作速度阈值：0.3 m/s
        }
    
    sensitivity_level = fall_new_strategy_sensitivity.currentText()
    
    if sensitivity_level == "灵敏":
        # 灵敏模式：较低的阈值，更容易检测到跌倒
        return {
            'time_window_ms': 2500,  # 时间窗口：2.5秒
            'height_threshold': 0.4,  # 高度下降阈值：0.4米
            'velocity_threshold': 0.4,  # 速度阈值：0.4 m/s
            'acceleration_threshold': 1.5,  # 加速度阈值：1.5 m/s²
            'low_height_threshold': 0.4,  # 低姿态高度阈值：0.4m
            'min_height_low': 0.6,  # 最低点阈值：0.6m
            'low_duration_threshold': 0.3,  # 低姿态持续时间阈值：0.3秒
            'min_conditions': 5,  # 最少满足条件数：5个（62.5%）
            'max_height_min': -2.0,  # 最大高度下限
            'max_height_max': 1.8,  # 最大高度上限
            'slow_velocity_threshold': 0.25  # 缓慢动作速度阈值：0.25 m/s
        }
    elif sensitivity_level == "中等":
        # 中等模式：平衡检测率和误报率
        return {
            'time_window_ms': 3000,  # 时间窗口：3秒
            'height_threshold': 0.5,  # 高度下降阈值：0.5米
            'velocity_threshold': 0.5,  # 速度阈值：0.5 m/s
            'acceleration_threshold': 2.0,  # 加速度阈值：2.0 m/s²
            'low_height_threshold': 0.3,  # 低姿态高度阈值：0.3m
            'min_height_low': 0.5,  # 最低点阈值：0.5m
            'low_duration_threshold': 0.5,  # 低姿态持续时间阈值：0.5秒
            'min_conditions': 6,  # 最少满足条件数：6个（75%）
            'max_height_min': -2.0,  # 最大高度下限
            'max_height_max': 1.5,  # 最大高度上限
            'slow_velocity_threshold': 0.3  # 缓慢动作速度阈值：0.3 m/s
        }
    else:  # "不灵敏"
        # 不灵敏模式：较高的阈值，减少误报但可能漏检
        return {
            'time_window_ms': 3500,  # 时间窗口：3.5秒
            'height_threshold': 0.6,  # 高度下降阈值：0.6米
            'velocity_threshold': 0.6,  # 速度阈值：0.6 m/s
            'acceleration_threshold': 2.5,  # 加速度阈值：2.5 m/s²
            'low_height_threshold': 0.2,  # 低姿态高度阈值：0.2m
            'min_height_low': 0.4,  # 最低点阈值：0.4m
            'low_duration_threshold': 0.7,  # 低姿态持续时间阈值：0.7秒
            'min_conditions': 7,  # 最少满足条件数：7个（87.5%）
            'max_height_min': -2.0,  # 最大高度下限
            'max_height_max': 1.2,  # 最大高度上限
            'slow_velocity_threshold': 0.35  # 缓慢动作速度阈值：0.35 m/s
        }

def update_figure():
    global img_rdi, img_rai, img_rti, img_rei, img_dti
    global idx,cnt
    global pointcloud_scatter, pointcloud_info_label, PointCloudData
    global pointcloud_max_points, pointcloud_refresh_rate, pointcloud_threshold
    global pointcloud_show_grid, pointcloud_show_axes
    global pointcloud_grid, pointcloud_grid_xy, pointcloud_grid_xz, pointcloud_grid_yz
    global pointcloud_x_axis, pointcloud_y_axis, pointcloud_z_axis
    global pointcloud_x_marker, pointcloud_y_marker, pointcloud_z_marker
    global pointcloud_last_update_time, pointcloud_refresh_interval
    global pointcloud_log_text, pointcloud_coordinate_info_label
    global PointCloudHistory
    global pointcloud_enable_clustering, pointcloud_cluster_eps, pointcloud_cluster_min_samples
    global trajectory_history, trajectory_plot, trajectory_info_label, trajectory_line_ref, trajectory_plot_widget
    global trajectory_history_length, trajectory_point_size, trajectory_show_axes, trajectory_show_grid
    global trajectory_view, trajectory_line_items, trajectory_scatter_items
    global target_count, trajectory_line_items, trajectory_scatter_items
    global height_history, height_plot, height_info_label, height_line_ref
    global height_history_length, height_point_size, height_show_axes, height_show_grid
    global fall_time_window, fall_height_threshold, fall_height_min, fall_height_max
    global fall_alert_label, last_fall_alert_time, fall_alert_cooldown
    global fall_detection_enabled, fall_new_strategy_enabled, fall_new_strategy_sensitivity

    # 时间-距离图绘制，向img_rti容器中添加需要绘制的数据，其中：
    # （1）RTIData.get()返回图像矩阵，图像由processor线程采集
    # （2）.sum(2)是对第三维求和（比如多个通道或Tx/Rx叠加）
    # （3）[0:1024:16,:]是抽稀处理，仅取部分数据减少渲染负担，16为稀疏化步长，已改成1
    # （4）img_rti.setImage(...)直接设置图像内容，立即生效
    # （5）使用QTimer.singleShot实现递归刷新机制，约每1ms调用一次，构成实时绘图。
    # （6）levels=[0, 1e4]是colorbar着色范围
    img_rti.setImage(RTIData.get().sum(2)[0:1024:1,:], levels=[0, 1e4])
    
    # 更新点云显示（带刷新速率控制）
    current_time = time.time() * 1000  # 转换为毫秒
    pointcloud_refresh_interval = pointcloud_refresh_rate.value()
    
    # 控制网格和坐标系的显示
    if pointcloud_show_grid.isChecked():
        # 显示所有三个平面的网格
        try:
            if 'pointcloud_grid_xy' in globals() and pointcloud_grid_xy is not None:
                pointcloud_grid_xy.show()
            if 'pointcloud_grid_xz' in globals() and pointcloud_grid_xz is not None:
                pointcloud_grid_xz.show()
            if 'pointcloud_grid_yz' in globals() and pointcloud_grid_yz is not None:
                pointcloud_grid_yz.show()
            # 兼容旧代码
            if 'pointcloud_grid' in globals() and pointcloud_grid is not None:
                pointcloud_grid.show()
        except:
            pass
    else:
        # 隐藏所有网格
        try:
            if 'pointcloud_grid_xy' in globals() and pointcloud_grid_xy is not None:
                pointcloud_grid_xy.hide()
            if 'pointcloud_grid_xz' in globals() and pointcloud_grid_xz is not None:
                pointcloud_grid_xz.hide()
            if 'pointcloud_grid_yz' in globals() and pointcloud_grid_yz is not None:
                pointcloud_grid_yz.hide()
            # 兼容旧代码
            if 'pointcloud_grid' in globals() and pointcloud_grid is not None:
                pointcloud_grid.hide()
        except:
            pass
    
    if pointcloud_show_axes.isChecked():
        pointcloud_x_axis.show()
        pointcloud_y_axis.show()
        pointcloud_z_axis.show()
        pointcloud_x_marker.show()
        pointcloud_y_marker.show()
        pointcloud_z_marker.show()
    else:
        pointcloud_x_axis.hide()
        pointcloud_y_axis.hide()
        pointcloud_z_axis.hide()
        pointcloud_x_marker.hide()
        pointcloud_y_marker.hide()
        pointcloud_z_marker.hide()
    
    # 更新阈值参数到全局变量（供DSP使用）
    gl.set_value('pointcloud_threshold', pointcloud_threshold.value())
    
    # 根据刷新速率决定是否更新点云
    if current_time - pointcloud_last_update_time >= pointcloud_refresh_interval:
        pointcloud_last_update_time = current_time
        
        # 清空队列中所有旧数据，只保留最新的
        queue_size_before = PointCloudData.qsize()
        while PointCloudData.qsize() > 1:
            try:
                PointCloudData.get_nowait()
            except:
                break
        
        # 调试信息：每100次更新打印一次
        if hasattr(update_figure, '_debug_counter'):
            update_figure._debug_counter += 1
        else:
            update_figure._debug_counter = 0
        
        if update_figure._debug_counter % 100 == 0:
            print(f"[点云显示] 队列大小: {queue_size_before} -> {PointCloudData.qsize()}")
        
        # 处理新点云数据并加入历史缓冲区
        if not PointCloudData.empty():
            try:
                new_pointcloud = PointCloudData.get()
                if new_pointcloud is not None and len(new_pointcloud) > 0:
                    # 将新点云加入历史缓冲区（带时间戳）
                    PointCloudHistory.append((current_time, new_pointcloud))
                    
                    # 限制历史缓冲区大小（最多保留最近20帧）
                    if len(PointCloudHistory) > 20:
                        PointCloudHistory.pop(0)
            except Exception as e:
                print(f"处理新点云数据错误: {e}")
        
        # 清理过期的点云（基于能量的时间衰减）
        # 近距离点（高能量，红色）：保留时间长（500ms）
        # 远距离点（低能量，蓝色）：保留时间短（100ms）
        valid_history = []
        for timestamp, pointcloud in PointCloudHistory:
            age_ms = current_time - timestamp
            
            # 根据点的距离计算保留时间
            # 距离越近（能量越高），保留时间越长
            if len(pointcloud) > 0:
                ranges = pointcloud[:, 0]
                # 计算该帧点云的平均距离
                avg_range = np.mean(ranges)
                
                # 根据平均距离设置保留时间
                # 近距离（<1m）：保留1000ms
                # 中距离（1-3m）：保留300ms
                # 远距离（>3m）：保留100ms
                if avg_range < 1.0:
                    max_age_ms = 1000  # 高能量点保留1000ms
                elif avg_range < 3.0:
                    max_age_ms = 300  # 中能量点保留300ms
                else:
                    max_age_ms = 100  # 低能量点保留100ms
                
                # 只保留未过期的点云
                if age_ms < max_age_ms:
                    valid_history.append((timestamp, pointcloud))
        
        PointCloudHistory = valid_history
        
        # 合并所有有效的历史点云（带时间信息）
        if len(PointCloudHistory) > 0:
            try:
                # 合并所有历史点云，同时记录每个点的时间信息
                all_points_with_time = []
                for timestamp, pointcloud in PointCloudHistory:
                    if pointcloud is not None and len(pointcloud) > 0:
                        # 为每个点添加时间戳信息
                        num_points = len(pointcloud)
                        timestamps = np.full((num_points, 1), timestamp)
                        # 扩展点云格式: [range, x, y, z, timestamp]
                        points_with_time = np.hstack([pointcloud, timestamps])
                        all_points_with_time.append(points_with_time)
                
                if len(all_points_with_time) > 0:
                    # 合并所有点云（格式: [range, x, y, z, timestamp]）
                    merged_pointcloud = np.vstack(all_points_with_time)
                    
                    # 提取数据
                    ranges = merged_pointcloud[:, 0]  # 距离
                    x = merged_pointcloud[:, 1]       # X坐标
                    y = merged_pointcloud[:, 2]       # Y坐标
                    z = merged_pointcloud[:, 3]       # Z坐标
                    timestamps = merged_pointcloud[:, 4]  # 时间戳
                    
                    # 如果启用了聚类功能，进行聚类处理
                    # 检查变量是否已初始化（可能在application()函数初始化之前调用）
                    enable_clustering = False
                    cluster_eps_val = 0.3
                    cluster_min_samples_val = 3
                    is_clustered = False  # 初始化聚类状态标记
                    
                    try:
                        if 'pointcloud_enable_clustering' in globals() and pointcloud_enable_clustering is not None:
                            enable_clustering = pointcloud_enable_clustering.isChecked()
                            if 'pointcloud_cluster_eps' in globals() and pointcloud_cluster_eps is not None:
                                cluster_eps_val = pointcloud_cluster_eps.value()
                            if 'pointcloud_cluster_min_samples' in globals() and pointcloud_cluster_min_samples is not None:
                                cluster_min_samples_val = pointcloud_cluster_min_samples.value()
                    except (NameError, AttributeError):
                        # 变量未初始化，使用默认值（不启用聚类）
                        enable_clustering = False
                    if enable_clustering:
                        # 准备点云数据（格式: [range, x, y, z]）
                        pointcloud_data = merged_pointcloud[:, :4]
                        
                        # 执行聚类（使用之前获取的参数）
                        clustered_pointcloud, cluster_labels, cluster_info = pointcloud_clustering.cluster_pointcloud_simple(
                            pointcloud_data, 
                            eps=cluster_eps_val, 
                            min_samples=cluster_min_samples_val
                        )
                        
                        # 更新点云数据
                        if len(clustered_pointcloud) > 0:
                            ranges = clustered_pointcloud[:, 0]
                            x = clustered_pointcloud[:, 1]
                            y = clustered_pointcloud[:, 2]
                            z = clustered_pointcloud[:, 3]
                            # 聚类后的点没有时间戳，使用当前时间
                            timestamps = np.full(len(clustered_pointcloud), current_time)
                            is_clustered = True  # 标记为已聚类
                            
                            # 输出聚类信息（每100次更新打印一次）
                            if update_figure._debug_counter % 100 == 0:
                                print(f"[点云聚类] 聚类前: {cluster_info['num_points_before']} 点 | "
                                      f"聚类后: {cluster_info['num_points_after']} 点 | "
                                      f"聚类数: {cluster_info['num_clusters']} | "
                                      f"噪声点: {cluster_info['num_noise']}")
                        else:
                            # 聚类后没有有效点，清空显示
                            ranges = np.array([])
                            x = np.array([])
                            y = np.array([])
                            z = np.array([])
                            timestamps = np.array([])
                            is_clustered = False
                    
                    # 限制点云数量（优先保留近距离的点）
                    max_points = pointcloud_max_points.value()
                    if len(x) > max_points:
                        # 按距离排序，优先保留近距离点（高能量点）
                        sorted_indices = np.argsort(ranges)
                        selected_indices = sorted_indices[:max_points]
                        ranges = ranges[selected_indices]
                        x = x[selected_indices]
                        y = y[selected_indices]
                        z = z[selected_indices]
                        timestamps = timestamps[selected_indices]
                    
                    # 改进的点云可视化：根据能量/幅值区分噪声点和目标点
                    # 噪声点：低能量（远距离），半透明浅蓝色，小尺寸
                    # 目标点：高能量（近距离/已聚类），橙色，大尺寸
                    if len(ranges) > 0:
                        # 计算能量阈值（使用距离作为能量代理）
                        # 近距离 = 高能量，远距离 = 低能量
                        energy_threshold = 2.0  # 2米作为能量阈值
                        
                        # 创建颜色数组和大小数组
                        color_array = np.zeros((len(ranges), 4))
                        size_array = np.zeros(len(ranges))
                        
                        for i in range(len(ranges)):
                            range_val = ranges[i]
                            age_ms = current_time - timestamps[i]
                            
                            # 判断是否为高能量点（目标点）
                            # 条件：近距离 OR 已聚类
                            is_target = (range_val < energy_threshold) or is_clustered
                            
                            if is_target:
                                # 目标点：橙色，不透明，大尺寸
                                # 橙色 RGB: (255, 165, 0) -> 归一化: (1.0, 0.65, 0.0)
                                color_array[i, 0] = 1.0   # R: 红色分量
                                color_array[i, 1] = 0  # G: 绿色分量
                                color_array[i, 2] = 0.0   # B: 蓝色分量
                                color_array[i, 3] = 0.7   # A: 完全不透明
                                size_array[i] = 9  # 大尺寸
                            else:
                                # 噪声点：浅蓝色，半透明，小尺寸
                                # 浅蓝色 RGB: (173, 216, 230) -> 归一化: (0.68, 0.85, 0.90)
                                color_array[i, 0] = 0.68  # R: 红色分量
                                color_array[i, 1] = 0.85  # G: 绿色分量
                                color_array[i, 2] = 0.90  # B: 蓝色分量
                                color_array[i, 3] = 0.3   # A: 半透明（30%透明度）
                                size_array[i] = 5  # 小尺寸
                            
                            # 根据时间衰减调整透明度（噪声点衰减更快）
                            if age_ms > 0:
                                if is_target:
                                    # 目标点：保留时间长，衰减慢
                                    max_age_ms = 1000 if range_val < 1.0 else 500
                                    if age_ms >= max_age_ms:
                                        color_array[i, 3] = 0.0
                                    else:
                                        decay_factor = 1.0 - (age_ms / max_age_ms)
                                        color_array[i, 3] *= max(0.7, decay_factor)
                                else:
                                    # 噪声点：保留时间短，快速衰减
                                    max_age_ms = 200
                                    if age_ms >= max_age_ms:
                                        color_array[i, 3] = 0.0
                                    else:
                                        decay_factor = 1.0 - (age_ms / max_age_ms)
                                        color_array[i, 3] *= decay_factor * 0.3
                    else:
                        # 如果没有点，创建空数组
                        color_array = np.ones((0, 4), dtype=np.float32) * 0.5  # 默认灰色
                        size_array = np.array([], dtype=np.float32)  # 空尺寸数组
                    
                    # 更新散点图
                    if len(x) > 0:
                        # 确保数据类型为 float32，pyqtgraph OpenGL 需要
                        pos = np.column_stack([x, y, z]).astype(np.float32)
                        # 确保 color_array 和 size_array 形状与 pos 匹配，并转换为 float32
                        if color_array.shape[0] != pos.shape[0]:
                            # 如果形状不匹配，创建默认颜色数组和尺寸数组
                            color_array = np.ones((pos.shape[0], 4), dtype=np.float32) * 0.5
                            size_array = np.full(pos.shape[0], 8, dtype=np.float32)
                        else:
                            color_array = color_array.astype(np.float32)
                            if len(size_array) != len(pos):
                                size_array = np.full(pos.shape[0], 8, dtype=np.float32)
                            else:
                                size_array = size_array.astype(np.float32)
                        
                        try:
                            # 使用动态大小数组（如果支持）或固定大小
                            # 注意：GLScatterPlotItem的size参数可以是标量或数组
                            if len(size_array) == len(pos) and len(size_array) > 0:
                                # 使用动态大小数组
                                pointcloud_scatter.setData(pos=pos, color=color_array, size=size_array)
                            else:
                                # 回退到固定大小
                                pointcloud_scatter.setData(pos=pos, color=color_array, size=8)
                        except Exception as e:
                            # 如果绘制失败，尝试清空并重新设置
                            print(f"点云绘制错误: {e}")
                            pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                                      color=np.empty((0, 4), dtype=np.float32), 
                                                      size=8)
                        
                        # 计算点云中心位置和方位信息（使用合并后的点云）
                        center_x = x.mean()
                        center_y = y.mean()
                        center_z = z.mean()
                        center_range = np.sqrt(center_x**2 + center_y**2 + center_z**2)
                    else:
                        # 如果没有点，清空散点图
                        try:
                            pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                                      color=np.empty((0, 4), dtype=np.float32), 
                                                      size=8)
                        except Exception as e:
                            print(f"清空点云错误: {e}")
                        # 设置默认值
                        center_x = center_y = center_z = center_range = 0.0
                    
                    # 计算方位角（azimuth，从X轴正方向逆时针，范围-180到180度）
                    azimuth_rad = np.arctan2(center_y, center_x)
                    azimuth_deg = np.rad2deg(azimuth_rad)
                    
                    # 计算俯仰角（elevation，从XY平面向上，范围-90到90度）
                    elevation_rad = np.arcsin(center_z / center_range) if center_range > 0.01 else 0
                    elevation_deg = np.rad2deg(elevation_rad)
                    
                    # 确定方位描述
                    def get_direction_description(azimuth_deg, elevation_deg):
                        """根据方位角和俯仰角返回方位描述"""
                        directions = []
                        
                        # 水平方位
                        if abs(azimuth_deg) < 22.5:
                            directions.append("前方")
                        elif abs(azimuth_deg - 180) < 22.5 or abs(azimuth_deg + 180) < 22.5:
                            directions.append("后方")
                        elif 67.5 < azimuth_deg < 112.5:
                            directions.append("左侧")
                        elif -112.5 < azimuth_deg < -67.5:
                            directions.append("右侧")
                        elif 22.5 <= azimuth_deg < 67.5:
                            directions.append("左前方")
                        elif 112.5 <= azimuth_deg < 157.5:
                            directions.append("左后方")
                        elif -157.5 <= azimuth_deg < -112.5:
                            directions.append("右后方")
                        elif -67.5 <= azimuth_deg < -22.5:
                            directions.append("右前方")
                        
                        # 垂直方位
                        if elevation_deg > 30:
                            directions.append("上方")
                        elif elevation_deg < -30:
                            directions.append("下方")
                        elif abs(elevation_deg) > 10:
                            if elevation_deg > 0:
                                directions.append("略上方")
                            else:
                                directions.append("略下方")
                        
                        return " ".join(directions) if directions else "中心"
                    
                    direction_desc = get_direction_description(azimuth_deg, elevation_deg)
                    
                    # 更新信息标签，显示四元组信息
                    if len(x) > 0:
                        range_info = f"距离范围: [{ranges.min():.2f}, {ranges.max():.2f}]m"
                        x_info = f"X范围: [{x.min():.2f}, {x.max():.2f}]m"
                        y_info = f"Y范围: [{y.min():.2f}, {y.max():.2f}]m"
                        z_info = f"Z范围: [{z.min():.2f}, {z.max():.2f}]m"
                        # 检查聚类状态（安全访问）
                        try:
                            clustering_status = " [已聚类]" if ('pointcloud_enable_clustering' in globals() and pointcloud_enable_clustering is not None and pointcloud_enable_clustering.isChecked()) else ""
                        except:
                            clustering_status = ""
                        pointcloud_info_label.setText(f"点云数量: {len(merged_pointcloud)} (显示: {len(x)}){clustering_status} | {range_info} | {x_info} | {y_info} | {z_info}")
                    else:
                        # 检查聚类状态（安全访问）
                        try:
                            clustering_status = " [已聚类]" if ('pointcloud_enable_clustering' in globals() and pointcloud_enable_clustering is not None and pointcloud_enable_clustering.isChecked()) else ""
                        except:
                            clustering_status = ""
                        pointcloud_info_label.setText(f"点云数量: {len(merged_pointcloud)} (显示: 0){clustering_status} | 无有效点云")
                    
                    # 更新坐标信息显示（已隐藏）
                    # coord_info = f"【IWR6843雷达坐标系】\n"
                    # coord_info += f"X轴(红色箭头) = 前方（雷达正对方向）\n"
                    # coord_info += f"Y轴(绿色箭头) = 左侧（方位向，8个阵元）\n"
                    # coord_info += f"Z轴(蓝色箭头) = 上方（俯仰向，2个阵元）\n"
                    # coord_info += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    # coord_info += f"点云中心位置: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f}) m\n"
                    # coord_info += f"距离雷达: {center_range:.2f} m\n"
                    # coord_info += f"方位角: {azimuth_deg:+.1f}° (0°=前方, +90°=左侧, -90°=右侧, ±180°=后方)\n"
                    # coord_info += f"俯仰角: {elevation_deg:+.1f}° (0°=水平, +90°=正上方, -90°=正下方)\n"
                    # coord_info += f"方位描述: {direction_desc}"
                    # if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                    #     pointcloud_coordinate_info_label.setText(coord_info)
                    
                    # 输出详细日志信息
                    log_msg = f"点云采集 | 总数: {len(merged_pointcloud)} | 显示: {len(x)} | "
                    log_msg += f"距离: [{ranges.min():.2f}, {ranges.max():.2f}]m | "
                    log_msg += f"X: [{x.min():.2f}, {x.max():.2f}]m | "
                    log_msg += f"Y: [{y.min():.2f}, {y.max():.2f}]m | "
                    log_msg += f"Z: [{z.min():.2f}, {z.max():.2f}]m"
                    print_pointcloud_log(log_msg, fontcolor='blue')
                    
                    # 初始化targets变量，确保高度检测可以访问
                    targets = []
                    
                    # 更新轨迹显示（找出能量最强的那一簇点云，加权计算中心）
                    try:
                        if len(x) > 0:
                            # 对当前帧的点云进行聚类，根据目标个数找出多个目标
                            current_frame_points = np.column_stack([ranges, x, y, z])
                            
                            # 获取目标个数
                            num_targets = target_count.value()
                            
                            # 使用聚类找出所有簇
                            from sklearn.cluster import DBSCAN
                            spatial_coords = np.column_stack([x, y, z])
                            clustering = DBSCAN(eps=0.3, min_samples=2)  # 使用较小的eps和min_samples
                            cluster_labels = clustering.fit_predict(spatial_coords)
                            
                            # 找出所有聚类的能量（能量用距离的倒数表示，距离越近能量越高）
                            unique_labels = np.unique(cluster_labels)
                            cluster_centers = []
                            
                            for label in unique_labels:
                                if label == -1:  # 跳过噪声点
                                    continue
                                
                                # 获取该聚类的所有点
                                cluster_mask = cluster_labels == label
                                cluster_points = current_frame_points[cluster_mask]
                                
                                if len(cluster_points) == 0:
                                    continue
                                
                                # 计算该聚类的总能量（使用距离的倒数）
                                cluster_ranges = cluster_points[:, 0]
                                cluster_energies = 1.0 / (cluster_ranges + 0.01)
                                cluster_total_energy = np.sum(cluster_energies)
                                
                                # 计算加权中心（权重为能量）
                                cluster_x = cluster_points[:, 1]
                                cluster_y = cluster_points[:, 2]
                                cluster_z = cluster_points[:, 3]
                                
                                # 归一化权重
                                weights = cluster_energies / cluster_energies.sum()
                                
                                # 计算加权中心
                                weighted_x = np.average(cluster_x, weights=weights)
                                weighted_y = np.average(cluster_y, weights=weights)
                                weighted_z = np.average(cluster_z, weights=weights)
                                weighted_range = np.sqrt(weighted_x**2 + weighted_y**2 + weighted_z**2)
                                
                                cluster_centers.append((cluster_total_energy, weighted_x, weighted_y, weighted_z, weighted_range))
                            
                            # 根据能量排序，选择前num_targets个目标
                            cluster_centers.sort(key=lambda x: x[0], reverse=True)  # 按能量降序排序
                            
                            # 获取目标中心列表
                            targets = []
                            if len(cluster_centers) > 0:
                                for i in range(min(num_targets, len(cluster_centers))):
                                    _, target_x, target_y, target_z, target_range = cluster_centers[i]
                                    if target_range > 0.01:  # 过滤无效目标
                                        targets.append((target_x, target_y, target_z, target_range))
                            
                            # 如果目标数量不足，使用整体中心补充
                            if len(targets) < num_targets:
                                targets.append((center_x, center_y, center_z, center_range))
                            
                            # 只保留前num_targets个目标
                            targets = targets[:num_targets]
                            
                            # 添加新的轨迹点（存储原始坐标，显示时只翻转Y轴）
                            # 轨迹历史存储格式: [(timestamp, target_id, x, y, range), ...]
                            # 注意：存储原始坐标，显示时Y轴会翻转，使第一象限在右下角
                            for target_id, (target_x, target_y, target_z, target_range) in enumerate(targets):
                                trajectory_history.append((current_time, target_id, target_x, target_y, target_range))
                            
                            # 限制轨迹历史长度
                            max_history = trajectory_history_length.value()
                            if len(trajectory_history) > max_history:
                                trajectory_history = trajectory_history[-max_history:]
                            
                            # 提取轨迹数据（支持多目标）
                            if len(trajectory_history) > 0:
                                # 按目标ID分组轨迹
                                num_targets = target_count.value()
                                target_colors = [
                                    (255, 0, 0, 200),      # 目标0: 红色
                                    (0, 255, 0, 200),      # 目标1: 绿色
                                    (0, 0, 255, 200),      # 目标2: 蓝色
                                    (255, 255, 0, 200),    # 目标3: 黄色
                                    (255, 0, 255, 200),    # 目标4: 洋红
                                    (0, 255, 255, 200),    # 目标5: 青色
                                    (255, 128, 0, 200),    # 目标6: 橙色
                                    (128, 0, 255, 200),    # 目标7: 紫色
                                    (255, 192, 203, 200),   # 目标8: 粉色
                                    (128, 128, 128, 200),  # 目标9: 灰色
                                ]
                                
                                # 收集所有轨迹点
                                all_traj_x = []
                                all_traj_y = []
                                all_traj_ranges = []
                                
                                for target_id in range(num_targets):
                                    # 获取该目标的所有轨迹点
                                    target_points = [t for t in trajectory_history if len(t) > 1 and t[1] == target_id]
                                    
                                    if len(target_points) > 0:
                                        traj_x = [t[2] for t in target_points]  # X坐标（原始值，X轴正向朝右）
                                        traj_y = [-t[3] for t in target_points]  # Y坐标（翻转，使Y轴正向朝下，第一象限在右下角）
                                        traj_ranges = [t[4] for t in target_points]  # 距离
                                        
                                        all_traj_x.extend(traj_x)
                                        all_traj_y.extend(traj_y)
                                        all_traj_ranges.extend(traj_ranges)
                                
                                if len(all_traj_x) > 0:
                                    # 更新轨迹散点图（2D显示，使用渐变色：根据距离从红色到蓝色）
                                    # 根据距离计算渐变色（近处红色，远处蓝色）
                                    all_traj_ranges_array = np.array(all_traj_ranges)
                                    if len(all_traj_ranges_array) > 0:
                                        # 归一化距离到0-1范围
                                        if all_traj_ranges_array.max() > all_traj_ranges_array.min():
                                            normalized_ranges = (all_traj_ranges_array - all_traj_ranges_array.min()) / (all_traj_ranges_array.max() - all_traj_ranges_array.min())
                                        else:
                                            normalized_ranges = np.zeros_like(all_traj_ranges_array)
                                        
                                        # 生成渐变色：红色(近) -> 蓝色(远)
                                        # R: 从255到0，G: 从0到0，B: 从0到255
                                        color_list = []
                                        for norm_range in normalized_ranges:
                                            r = int(255 * (1 - norm_range))
                                            g = 0
                                            b = int(255 * norm_range)
                                            color_list.append((r, g, b, 200))  # 添加透明度
                                    else:
                                        # 如果没有距离数据，使用默认红色
                                        color_list = [(255, 0, 0, 200)] * len(all_traj_x)
                                    
                                    # 更新2D散点图（使用x和y坐标，颜色列表）
                                    trajectory_plot.setData(x=all_traj_x, y=all_traj_y, brush=color_list, size=trajectory_point_size.value())
                                    
                                    # 更新轨迹连线（显示所有点的连线）
                                    if len(all_traj_x) > 1:
                                        trajectory_line_ref.setData(x=all_traj_x, y=all_traj_y)
                                    else:
                                        trajectory_line_ref.setData([], [])
                                    
                                    # 清除旧的连线（不再需要，因为使用统一的连线）
                                    for line_item in trajectory_line_items:
                                        trajectory_plot_widget.removeItem(line_item)
                                    for scatter_item in trajectory_scatter_items:
                                        trajectory_plot_widget.removeItem(scatter_item)
                                    trajectory_line_items = []
                                    trajectory_scatter_items = []
                                
                                # 更新轨迹信息
                                if len(trajectory_history) > 0:
                                    # 获取最新一帧的所有目标位置
                                    latest_timestamp = max([t[0] for t in trajectory_history])
                                    latest_targets = [t for t in trajectory_history if t[0] == latest_timestamp]
                                    
                                    traj_info = f"目标数: {num_targets} | 轨迹点数: {len(trajectory_history)} | "
                                    if len(latest_targets) > 0:
                                        traj_info += "最新位置: "
                                        for i, t in enumerate(latest_targets[:num_targets]):
                                            target_id = t[1] if len(t) > 1 else i
                                            traj_x_original = t[2] if len(t) > 2 else 0  # X坐标（原始值）
                                            traj_y_original = t[3] if len(t) > 3 else 0  # Y坐标（原始值）
                                            traj_range = t[4] if len(t) > 4 else 0
                                            traj_info += f"目标{target_id}: ({traj_x_original:.2f}, {traj_y_original:.2f}) m "
                                        trajectory_info_label.setText(traj_info)
                                else:
                                    trajectory_info_label.setText("轨迹信息: 等待数据...")
                            else:
                                trajectory_plot.setData([], [])
                                trajectory_line_ref.setData([], [])
                                # 清除所有连线和散点图
                                if 'trajectory_line_items' in globals():
                                    for line_item in trajectory_line_items:
                                        if 'trajectory_plot_widget' in globals():
                                            trajectory_plot_widget.removeItem(line_item)
                                    trajectory_line_items = []
                                if 'trajectory_scatter_items' in globals():
                                    for scatter_item in trajectory_scatter_items:
                                        if 'trajectory_plot_widget' in globals():
                                            trajectory_plot_widget.removeItem(scatter_item)
                                    trajectory_scatter_items = []
                                trajectory_info_label.setText("轨迹信息: 等待数据...")
                            
                    except Exception as e:
                        # 轨迹更新失败不影响主流程
                        if 'trajectory_history' in globals():
                            pass
                        # 如果轨迹更新失败，targets可能为空，使用点云中心作为备选
                        if len(targets) == 0 and len(x) > 0:
                            # 使用点云中心作为目标
                            targets = [(center_x, center_y, center_z, center_range)]
                    
                    # 更新高度检测（Z-Y轴，Z轴朝上）- 独立于轨迹更新
                    # 确保targets存在且有数据
                    if 'height_history' in globals() and 'height_plot' in globals() and len(x) > 0:
                        # 如果targets为空，尝试使用点云中心
                        if len(targets) == 0:
                            if len(x) > 0:
                                targets = [(center_x, center_y, center_z, center_range)]
                        
                        if len(targets) > 0:
                            try:
                                # 使用第一个目标的高度数据（或者可以扩展为多目标）
                                target_x, target_y, target_z, target_range = targets[0]
                                
                                # 异常点检测和过滤函数
                                def is_height_outlier(new_z, new_y, new_range, history, max_change_rate=0.5, max_height_range=2.0, max_y_range=2.0):
                                    """
                                    检测高度点是否为异常点（包括Z轴和Y轴）
                                    
                                    Args:
                                        new_z: 新的Z坐标（高度）
                                        new_y: 新的Y坐标（水平方向）
                                        new_range: 新的距离
                                        history: 历史高度数据列表，格式: [(timestamp, z, y, range), ...]
                                        max_change_rate: 最大变化率（相对于距离的比例）
                                        max_height_range: 最大合理高度范围（米）
                                        max_y_range: 最大合理水平方向范围（米）
                                    
                                    Returns:
                                        bool: True表示是异常点，False表示正常点
                                    """
                                    if len(history) == 0:
                                        # 第一个点，检查是否在合理范围内
                                        if abs(new_z) > max_height_range:
                                            return True
                                        if abs(new_y) > max_y_range:
                                            return True
                                        return False
                                    
                                    # 方法1: 检查高度和水平方向是否在合理范围内
                                    if abs(new_z) > max_height_range:
                                        return True
                                    if abs(new_y) > max_y_range:
                                        return True
                                    
                                    # 方法2: 检查相邻点之间的变化率（如果历史数据足够）
                                    if len(history) >= 1:
                                        last_z = history[-1][1]
                                        last_y = history[-1][2]
                                        last_range = history[-1][3]
                                        
                                        # 计算高度变化和水平方向变化
                                        height_change = abs(new_z - last_z)
                                        y_change = abs(new_y - last_y)
                                        
                                        # 如果距离变化不大，但高度或水平方向变化很大，可能是异常点
                                        range_change = abs(new_range - last_range)
                                        if range_change < 0.1:  # 距离变化小于10cm
                                            # 高度变化不应超过一定阈值
                                            # 距离越近，阈值越小（更严格）
                                            if new_range < 0.5:
                                                threshold_z = 0.15  # 距离<0.5m时，阈值0.15m
                                                threshold_y = 0.2   # 水平方向阈值0.2m
                                            elif new_range < 1.0:
                                                threshold_z = 0.25  # 距离<1.0m时，阈值0.25m
                                                threshold_y = 0.3   # 水平方向阈值0.3m
                                            else:
                                                threshold_z = 0.3   # 距离>=1.0m时，阈值0.3m
                                                threshold_y = 0.4   # 水平方向阈值0.4m
                                            
                                            if height_change > threshold_z:
                                                return True
                                            if y_change > threshold_y:
                                                return True
                                        
                                        # 如果距离很近（<1米），高度和水平方向变化不应过大
                                        if new_range < 1.0:
                                            # 距离越近，阈值越小
                                            if new_range < 0.5:
                                                max_change_z = 0.3  # 距离<0.5m时，最大变化0.3m
                                                max_change_y = 0.4  # 水平方向最大变化0.4m
                                            else:
                                                max_change_z = 0.4  # 距离0.5-1.0m时，最大变化0.4m
                                                max_change_y = 0.5  # 水平方向最大变化0.5m
                                            if height_change > max_change_z:
                                                return True
                                            if y_change > max_change_y:
                                                return True
                                    
                                    # 方法3: 使用IQR方法检测离群点（如果历史数据足够）
                                    if len(history) >= 5:
                                        recent_heights = [h[1] for h in history[-10:]]  # 使用最近10个点的Z值
                                        recent_y_values = [h[2] for h in history[-10:]]  # 使用最近10个点的Y值
                                        recent_heights_array = np.array(recent_heights)
                                        recent_y_array = np.array(recent_y_values)
                                        
                                        # 检测Z轴离群点
                                        q1_z = np.percentile(recent_heights_array, 25)
                                        q3_z = np.percentile(recent_heights_array, 75)
                                        iqr_z = q3_z - q1_z
                                        
                                        # 检测Y轴离群点
                                        q1_y = np.percentile(recent_y_array, 25)
                                        q3_y = np.percentile(recent_y_array, 75)
                                        iqr_y = q3_y - q1_y
                                        
                                        # 如果IQR太小（数据太集中），使用固定阈值
                                        if iqr_z < 0.1:
                                            # 使用中位数和固定阈值（Z轴）
                                            median_height = np.median(recent_heights_array)
                                            if new_range < 0.5:
                                                threshold = 0.2  # 距离<0.5m时，阈值0.2m
                                            elif new_range < 1.0:
                                                threshold = 0.3  # 距离<1.0m时，阈值0.3m
                                            else:
                                                threshold = 0.5  # 距离>=1.0m时，阈值0.5m
                                            
                                            if abs(new_z - median_height) > threshold:
                                                return True
                                        else:
                                            # 使用IQR方法（Z轴）
                                            # 距离越近，使用更严格的倍数
                                            if new_range < 0.5:
                                                multiplier = 1.0  # 距离<0.5m时，使用1倍IQR
                                            elif new_range < 1.0:
                                                multiplier = 1.5  # 距离<1.0m时，使用1.5倍IQR
                                            else:
                                                multiplier = 2.0  # 距离>=1.0m时，使用2倍IQR
                                            
                                            lower_bound_z = q1_z - multiplier * iqr_z
                                            upper_bound_z = q3_z + multiplier * iqr_z
                                            if new_z < lower_bound_z or new_z > upper_bound_z:
                                                return True
                                        
                                        # 检测Y轴离群点
                                        if iqr_y < 0.1:
                                            # 使用中位数和固定阈值（Y轴）
                                            median_y = np.median(recent_y_array)
                                            if new_range < 0.5:
                                                threshold_y = 0.3  # 距离<0.5m时，阈值0.3m
                                            elif new_range < 1.0:
                                                threshold_y = 0.4  # 距离<1.0m时，阈值0.4m
                                            else:
                                                threshold_y = 0.5  # 距离>=1.0m时，阈值0.5m
                                            
                                            if abs(new_y - median_y) > threshold_y:
                                                return True
                                        else:
                                            # 使用IQR方法（Y轴）
                                            # 水平方向使用更严格的倍数
                                            if new_range < 0.5:
                                                multiplier_y = 1.0  # 距离<0.5m时，使用1倍IQR
                                            elif new_range < 1.0:
                                                multiplier_y = 1.2  # 距离<1.0m时，使用1.2倍IQR
                                            else:
                                                multiplier_y = 1.5  # 距离>=1.0m时，使用1.5倍IQR
                                            
                                            lower_bound_y = q1_y - multiplier_y * iqr_y
                                            upper_bound_y = q3_y + multiplier_y * iqr_y
                                            if new_y < lower_bound_y or new_y > upper_bound_y:
                                                return True
                                    
                                    return False
                                
                                # 检测新点是否为异常点
                                is_outlier = is_height_outlier(target_z, -target_y, target_range, height_history)
                                
                                if not is_outlier:
                                    # 正常点，添加到历史记录
                                    height_history.append((current_time, target_z, -target_y, target_range))
                                else:
                                    # 异常点，跳过或使用中位数平滑
                                    if len(height_history) >= 2:
                                        # 使用最近几个点的中位数作为平滑值
                                        recent_z = [h[1] for h in height_history[-3:]]
                                        recent_y = [h[2] for h in height_history[-3:]]
                                        smoothed_z = np.median(recent_z)
                                        smoothed_y = np.median(recent_y)
                                        # 如果平滑后的值也在合理范围内（Z轴和Y轴都要检查），使用平滑值
                                        if abs(smoothed_z) < 2.0 and abs(smoothed_y) < 2.0:
                                            height_history.append((current_time, smoothed_z, smoothed_y, target_range))
                                        # 否则完全跳过这个点
                                    # 调试信息（每100次打印一次）
                                    if update_figure._debug_counter % 100 == 0:
                                        print(f"[高度检测] 过滤异常点: Z={target_z:.2f}m, Y={target_y:.2f}m, 距离={target_range:.2f}m")
                                
                                # 限制高度历史长度
                                max_height_history = height_history_length.value()
                                if len(height_history) > max_height_history:
                                    height_history = height_history[-max_height_history:]
                                
                                # 提取高度数据（过滤后的）
                                if len(height_history) > 0:
                                    height_z = [h[1] for h in height_history]
                                    height_y = [h[2] for h in height_history]
                                    height_ranges = [h[3] for h in height_history]
                                    
                                    # 再次使用IQR方法过滤显示数据中的离群点（Z轴和Y轴都过滤）
                                    if len(height_z) >= 5:
                                        height_z_array = np.array(height_z)
                                        height_y_array = np.array(height_y)
                                        
                                        # Z轴（高度）离群点检测
                                        q1_z = np.percentile(height_z_array, 25)
                                        q3_z = np.percentile(height_z_array, 75)
                                        iqr_z = q3_z - q1_z
                                        
                                        # Y轴（水平方向）离群点检测
                                        q1_y = np.percentile(height_y_array, 25)
                                        q3_y = np.percentile(height_y_array, 75)
                                        iqr_y = q3_y - q1_y
                                        
                                        # 创建有效点掩码（Z轴和Y轴都要满足条件）
                                        valid_mask = np.ones(len(height_z), dtype=bool)
                                        
                                        # Z轴过滤
                                        if iqr_z > 0.05:  # 只有当IQR足够大时才过滤
                                            lower_bound_z = q1_z - 1.5 * iqr_z
                                            upper_bound_z = q3_z + 1.5 * iqr_z
                                            valid_mask = valid_mask & (height_z_array >= lower_bound_z) & (height_z_array <= upper_bound_z)
                                        
                                        # Y轴过滤（水平方向）
                                        if iqr_y > 0.05:  # 只有当IQR足够大时才过滤
                                            lower_bound_y = q1_y - 1.2 * iqr_y  # Y轴使用更严格的倍数
                                            upper_bound_y = q3_y + 1.2 * iqr_y
                                            valid_mask = valid_mask & (height_y_array >= lower_bound_y) & (height_y_array <= upper_bound_y)
                                        
                                        # 过滤数据
                                        height_z = [height_z[i] for i in range(len(height_z)) if valid_mask[i]]
                                        height_y = [height_y[i] for i in range(len(height_y)) if valid_mask[i]]
                                        height_ranges = [height_ranges[i] for i in range(len(height_ranges)) if valid_mask[i]]
                                    
                                    # 更新高度散点图（根据距离设置颜色，近处红色，远处蓝色）
                                    if len(height_ranges) > 0 and max(height_ranges) > min(height_ranges):
                                        height_colors = (np.array(height_ranges) - min(height_ranges)) / (max(height_ranges) - min(height_ranges))
                                        height_color_list = []
                                        for c in height_colors:
                                            height_color_list.append((int(255 * (1 - c)), 0, int(255 * c), 200))  # 红色到蓝色
                                    else:
                                        height_color_list = [(255, 0, 0, 200)] * len(height_z)
                                    
                                    # 更新高度点（Z轴朝上：X轴显示Y值，Y轴显示Z值）
                                    # 注意：Y轴已翻转，所以x=height_y（水平），y=height_z（垂直，朝上）
                                    height_plot.setData(x=height_y, y=height_z, brush=height_color_list, size=height_point_size.value())
                                    
                                    # 更新高度连线
                                    if len(height_y) > 1:
                                        height_line_ref.setData(x=height_y, y=height_z)
                                    else:
                                        height_line_ref.setData([], [])
                                    
                                    # 更新高度信息（显示原始Y坐标，注意高度中Y已翻转）
                                    if len(height_z) > 0:
                                        current_height_y_original = -height_y[-1] if len(height_y) > 0 else 0  # 翻转回来显示
                                        current_height_z = height_z[-1] if len(height_z) > 0 else 0
                                        current_height_range = height_ranges[-1] if len(height_ranges) > 0 else 0
                                        
                                        height_info = f"高度点数: {len(height_z)} | "
                                        height_info += f"当前位置: Z={current_height_z:.2f} m, Y={current_height_y_original:.2f} m | "
                                        height_info += f"距离: {current_height_range:.2f} m | "
                                        height_info += f"Z范围: [{min(height_z):.2f}, {max(height_z):.2f}] m | "
                                        # Y范围需要翻转回来显示
                                        height_y_original = [-y for y in height_y]
                                        height_info += f"Y范围: [{min(height_y_original):.2f}, {max(height_y_original):.2f}] m"
                                        height_info_label.setText(height_info)
                                        
                                        # 跌倒检测逻辑（只有当跌倒检测开关打开时才执行）
                                        if ('fall_detection_enabled' in globals() and fall_detection_enabled.isChecked() and 
                                            'fall_time_window' in globals() and 'fall_height_threshold' in globals() and 
                                            len(height_history) >= 2):
                                            try:
                                                # 获取检测参数
                                                time_window_ms = fall_time_window.value() * 1000  # 转换为毫秒
                                                height_threshold = fall_height_threshold.value()  # 高度下降阈值（米）
                                                height_window_min = fall_height_min.value()  # 高度检测窗口 - 最小高度（米）
                                                height_window_max = fall_height_max.value()  # 高度检测窗口 - 最大高度（米）
                                                
                                                # 查找时间窗口内的数据点
                                                recent_points = []
                                                for h in height_history:
                                                    age_ms = current_time - h[0]
                                                    if age_ms <= time_window_ms:
                                                        recent_points.append(h)
                                                
                                                if len(recent_points) >= 2:
                                                    # 检查是否启用新策略
                                                    use_new_strategy = ('fall_new_strategy_enabled' in globals() and 
                                                                       fall_new_strategy_enabled.isChecked())
                                                    
                                                    if use_new_strategy:
                                                        # ========== 新策略：多特征融合跌倒检测 ==========
                                                        # 根据选择的灵敏度等级获取参数
                                                        params = get_new_strategy_params()
                                                        NEW_STRATEGY_TIME_WINDOW_MS = params['time_window_ms']
                                                        NEW_STRATEGY_HEIGHT_THRESHOLD = params['height_threshold']
                                                        VELOCITY_THRESHOLD = params['velocity_threshold']
                                                        ACCELERATION_THRESHOLD = params['acceleration_threshold']
                                                        LOW_HEIGHT_THRESHOLD = params['low_height_threshold']
                                                        MIN_HEIGHT_LOW = params['min_height_low']
                                                        LOW_DURATION_THRESHOLD = params['low_duration_threshold']
                                                        MIN_CONDITIONS = params['min_conditions']
                                                        MAX_HEIGHT_MIN = params['max_height_min']
                                                        MAX_HEIGHT_MAX = params['max_height_max']
                                                        SLOW_VELOCITY_THRESHOLD = params['slow_velocity_threshold']
                                                        
                                                        # 查找新策略时间窗口内的数据点
                                                        new_strategy_points = []
                                                        for h in height_history:
                                                            age_ms = current_time - h[0]
                                                            if age_ms <= NEW_STRATEGY_TIME_WINDOW_MS:
                                                                new_strategy_points.append(h)
                                                        
                                                        if len(new_strategy_points) >= 2:
                                                            # 提取数据
                                                            recent_heights = [h[1] for h in new_strategy_points]  # Z坐标
                                                            recent_y_values = [h[2] for h in new_strategy_points]  # Y坐标（水平方向）
                                                            recent_ranges = [h[3] for h in new_strategy_points]  # 距离
                                                            recent_timestamps = [h[0] for h in new_strategy_points]  # 时间戳
                                                            
                                                            # 找到最高点和最低点
                                                            max_height_idx = np.argmax(recent_heights)
                                                            min_height_idx = np.argmin(recent_heights)
                                                            
                                                            # 确保最低点在最高点之后（时间顺序）
                                                            if recent_timestamps[min_height_idx] > recent_timestamps[max_height_idx]:
                                                                max_height = recent_heights[max_height_idx]
                                                                min_height = recent_heights[min_height_idx]
                                                                height_drop = max_height - min_height
                                                                time_duration = (recent_timestamps[min_height_idx] - recent_timestamps[max_height_idx]) / 1000.0  # 转换为秒
                                                                
                                                                # 阶段1：快速下降检测（使用新策略的固定阈值）
                                                                if height_drop >= NEW_STRATEGY_HEIGHT_THRESHOLD and time_duration > 0:
                                                                    # 计算下降速度（m/s）
                                                                    drop_velocity = height_drop / time_duration
                                                                    
                                                                    # 阶段2：速度验证（下降速度必须足够快）
                                                                    if drop_velocity > VELOCITY_THRESHOLD:
                                                                        # 计算加速度（需要至少3个点）
                                                                        acceleration = 0.0
                                                                        if len(new_strategy_points) >= 3:
                                                                            # 计算最近几个点的速度变化
                                                                            velocities = []
                                                                            for i in range(1, len(new_strategy_points)):
                                                                                dt = (recent_timestamps[i] - recent_timestamps[i-1]) / 1000.0
                                                                                if dt > 0:
                                                                                    dz = recent_heights[i] - recent_heights[i-1]
                                                                                    velocities.append(dz / dt)
                                                                            
                                                                            if len(velocities) >= 2:
                                                                                # 计算加速度（速度变化率）
                                                                                dt_accel = (recent_timestamps[-1] - recent_timestamps[-2]) / 1000.0
                                                                                if dt_accel > 0:
                                                                                    acceleration = (velocities[-1] - velocities[-2]) / dt_accel
                                                                        else:
                                                                            # 如果点不够，使用平均加速度估算
                                                                            acceleration = drop_velocity / time_duration
                                                                    
                                                                    # 阶段3：加速度验证（加速度必须足够大）
                                                                    if acceleration > ACCELERATION_THRESHOLD:
                                                                        # 计算水平位移
                                                                        y_change = abs(recent_y_values[min_height_idx] - recent_y_values[max_height_idx])
                                                                        
                                                                        # 检查最终高度（跌倒后应该稳定在低值）
                                                                        final_height = min_height
                                                                        
                                                                        # 检查低姿态持续时间（跌倒后应该保持低姿态）
                                                                        low_height_duration = 0.0
                                                                        for i in range(min_height_idx, len(new_strategy_points)):
                                                                            if recent_heights[i] < LOW_HEIGHT_THRESHOLD:
                                                                                if i < len(new_strategy_points) - 1:
                                                                                    low_height_duration += (recent_timestamps[i+1] - recent_timestamps[i]) / 1000.0
                                                                                else:
                                                                                    low_height_duration += (current_time - recent_timestamps[i]) / 1000.0
                                                                        
                                                                        # 阶段4：多条件组合判断（根据灵敏度等级使用不同阈值）
                                                                        conditions = {
                                                                            'height_drop': height_drop >= NEW_STRATEGY_HEIGHT_THRESHOLD,
                                                                            'velocity': drop_velocity > VELOCITY_THRESHOLD,
                                                                            'acceleration': acceleration > ACCELERATION_THRESHOLD,
                                                                            'final_height': final_height < LOW_HEIGHT_THRESHOLD,
                                                                            'height_window': MAX_HEIGHT_MIN <= max_height <= MAX_HEIGHT_MAX,
                                                                            'max_height_positive': max_height > 0,
                                                                            'min_height_low': min_height < MIN_HEIGHT_LOW,
                                                                            'low_duration': low_height_duration > LOW_DURATION_THRESHOLD
                                                                        }
                                                                        
                                                                        # 排除误报：缓慢动作
                                                                        if drop_velocity < SLOW_VELOCITY_THRESHOLD:  # 速度太慢，可能是下蹲或坐下
                                                                            conditions['velocity'] = False
                                                                        
                                                                        # 排除误报：高度恢复（检测到下降后高度快速恢复）
                                                                        if len(new_strategy_points) >= 3:
                                                                            # 检查最后几个点是否高度恢复
                                                                            last_heights = recent_heights[-3:]
                                                                            if len(last_heights) >= 2:
                                                                                height_recovery = last_heights[-1] - last_heights[0]
                                                                                if height_recovery > 0.2:  # 高度恢复超过0.2m，可能是误报
                                                                                    conditions['final_height'] = False
                                                                        
                                                                        # 计算满足条件的数量
                                                                        satisfied_conditions = sum(conditions.values())
                                                                        total_conditions = len(conditions)
                                                                        
                                                                        # 根据灵敏度等级，需要满足不同数量的条件才判定为跌倒
                                                                        if satisfied_conditions >= MIN_CONDITIONS:
                                                                            # 检查冷却时间，避免重复提示
                                                                            if current_time - last_fall_alert_time > fall_alert_cooldown:
                                                                                # 显示跌倒提示
                                                                                if 'fall_alert_label' in globals() and fall_alert_label is not None:
                                                                                    fall_alert_label.setText("你跌倒了！")
                                                                                    fall_alert_label.show()
                                                                                    
                                                                                    # 3秒后隐藏提示
                                                                                    QtCore.QTimer.singleShot(3000, lambda: fall_alert_label.hide() if 'fall_alert_label' in globals() and fall_alert_label is not None else None)
                                                                                
                                                                                # 播放音效
                                                                                try:
                                                                                    import winsound
                                                                                    # 播放系统警告音（频率1000Hz，持续500ms）
                                                                                    winsound.Beep(1000, 500)
                                                                                except Exception as e:
                                                                                    print(f"音效播放失败: {e}")
                                                                                
                                                                                # 更新最后提示时间
                                                                                last_fall_alert_time = current_time
                                                                                sensitivity_level = fall_new_strategy_sensitivity.currentText()
                                                                                print(f"[新策略跌倒检测] 检测到跌倒！(灵敏度: {sensitivity_level})")
                                                                                print(f"  高度: {max_height:.2f}m → {min_height:.2f}m (下降 {height_drop:.2f}m)")
                                                                                print(f"  速度: {drop_velocity:.2f} m/s, 加速度: {acceleration:.2f} m/s²")
                                                                                print(f"  水平位移: {y_change:.2f}m, 低姿态持续时间: {low_height_duration:.2f}s")
                                                                                print(f"  满足条件: {satisfied_conditions}/{total_conditions} (需要至少{MIN_CONDITIONS}个)")
                                                    else:
                                                        # ========== 旧策略：简单高度下降检测 ==========
                                                        # 找到时间窗口内的最高点和最低点
                                                        recent_heights = [h[1] for h in recent_points]  # Z坐标
                                                        recent_timestamps = [h[0] for h in recent_points]
                                                        
                                                        max_height_idx = np.argmax(recent_heights)
                                                        min_height_idx = np.argmin(recent_heights)
                                                        
                                                        # 确保最低点在最高点之后（时间顺序）
                                                        if recent_timestamps[min_height_idx] > recent_timestamps[max_height_idx]:
                                                            max_height = recent_heights[max_height_idx]
                                                            min_height = recent_heights[min_height_idx]
                                                            height_drop = max_height - min_height
                                                            
                                                            # 检测跌倒：高度快速下降到负值
                                                            # 条件1：高度下降超过阈值
                                                            # 条件2：最低点必须是负值（从正值快速降到0不算跌倒）
                                                            # 条件3：最高点必须是正值（确保是从高处跌落的）
                                                            # 条件4：最高点必须在高度检测窗口内（height_window_min <= max_height <= height_window_max）
                                                            if (height_drop >= height_threshold and 
                                                                min_height < 0 and 
                                                                max_height > 0 and
                                                                height_window_min <= max_height <= height_window_max):
                                                                # 检查冷却时间，避免重复提示
                                                                if current_time - last_fall_alert_time > fall_alert_cooldown:
                                                                    # 显示跌倒提示
                                                                    if 'fall_alert_label' in globals() and fall_alert_label is not None:
                                                                        fall_alert_label.setText("你跌倒了！")
                                                                        fall_alert_label.show()
                                                                        
                                                                        # 3秒后隐藏提示
                                                                        QtCore.QTimer.singleShot(3000, lambda: fall_alert_label.hide() if 'fall_alert_label' in globals() and fall_alert_label is not None else None)
                                                                    
                                                                    # 播放音效
                                                                    try:
                                                                        import winsound
                                                                        # 播放系统警告音（频率1000Hz，持续500ms）
                                                                        winsound.Beep(1000, 500)
                                                                    except Exception as e:
                                                                        print(f"音效播放失败: {e}")
                                                                    
                                                                    # 更新最后提示时间
                                                                    last_fall_alert_time = current_time
                                                                    print(f"[跌倒检测] 检测到跌倒！高度从 {max_height:.2f}m 下降到 {min_height:.2f}m，下降 {height_drop:.2f}m（高度窗口: {height_window_min:.2f}m ~ {height_window_max:.2f}m）")
                                            except Exception as e:
                                                # 跌倒检测失败不影响主流程
                                                if update_figure._debug_counter % 100 == 0:
                                                    print(f"跌倒检测错误: {e}")
                            except Exception as e:
                                # 高度更新失败不影响主流程
                                print(f"高度检测更新错误: {e}")
                                import traceback
                                traceback.print_exc()
                                if 'height_history' in globals():
                                    pass
                else:
                    # 历史缓冲区为空，清空显示
                    try:
                        pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                                  color=np.empty((0, 4), dtype=np.float32), 
                                                  size=8)
                    except Exception as e:
                        print(f"清空点云错误: {e}")
                    pointcloud_info_label.setText("点云数量: 0 | 等待数据...")
                    if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                        pointcloud_coordinate_info_label.setText("等待点云数据...")
            except Exception as e:
                print(f"点云显示错误: {e}")
                import traceback
                traceback.print_exc()
                if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                    pointcloud_coordinate_info_label.setText("点云处理错误，请查看日志")
        else:
            # 历史缓冲区为空时，清空显示
            if len(PointCloudHistory) == 0:
                try:
                    pointcloud_scatter.setData(pos=np.empty((0, 3), dtype=np.float32), 
                                              color=np.empty((0, 4), dtype=np.float32), 
                                              size=8)
                except Exception as e:
                    print(f"清空点云错误: {e}")
            # 队列为空时也更新信息，显示队列状态
            if update_figure._debug_counter % 200 == 0:  # 每200次更新打印一次
                print(f"[点云显示] 队列为空，等待数据...")
                print_pointcloud_log("队列状态: 等待数据...", fontcolor='orange')
            pointcloud_info_label.setText(f"点云数量: 0 | 队列状态: 等待数据...")
            if 'pointcloud_coordinate_info_label' in globals() and pointcloud_coordinate_info_label is not None:
                pointcloud_coordinate_info_label.setText("等待点云数据...")

    # img_rdi.setImage(RDIData.get()[:, :, 0].T, levels=[30, 50])
    img_rdi.setImage(RDIData.get().sum(0)[:, :, 0].T,levels=[2e4, 4e5])
    # img_rei.setImage(REIData.get().T,levels=[0, 3])
    img_rei.setImage(REIData.get()[4:12,:,:].sum(0).T,levels=[0, 8])
    img_dti.setImage(DTIData.get(),levels=[0, 1000])
    # img_rai.setImage(RAIData.get().sum(0).T, levels=[1.2e3, 4e6])
    # img_rai.setImage(RAIData.get()[0,:,:].T, levels=[8e3, 2e4])
    # img_rai.setImage(RAIData.get(),levels=[0, 3])
    img_rai.setImage(RAIData.get()[4:12,:,:].sum(0),levels=[0, 8])


    if gl.get_value('usr_gesture'):
        RT_feature = RTIData.get().sum(2)[0:1024:16,:]
        DT_feature = DTIData.get()
        RDT_feature = RDIData.get()[:, :, :, 0]
        ART_feature = RAIData.get()
        ERT_feature = REIData.get()

        # if Recognizebtn.isChecked():
        if Recognizebtn.isChecked():
            # 识别
            
            time_start = time.time()  # 记录开始时间
            result = Judge_gesture(RT_feature,DT_feature,RDT_feature,
                                            ART_feature,ERT_feature)
            time_end = time.time()  # 记录结束时间
            time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
            printlog('识别时间:'+str(time_sum)+'s, '+'识别结果:'+result,fontcolor='blue')


        elif CaptureDatabtn.isChecked() and datasetsencefile != '':
            idx=idx+1
            # 收集
            np.save(datasetsencefile+'/RT_feature_'+str(idx).zfill(5)+'.npy',RT_feature)
            np.save(datasetsencefile+'/DT_feature_'+str(idx).zfill(5)+'.npy',DT_feature)
            np.save(datasetsencefile+'/RDT_feature_'+str(idx).zfill(5)+'.npy',RDT_feature)
            np.save(datasetsencefile+'/ART_feature_'+str(idx).zfill(5)+'.npy',ART_feature)
            np.save(datasetsencefile+'/ERT_feature_'+str(idx).zfill(5)+'.npy',ERT_feature)
            printlog('采集到特征:'+datasetfilebox.currentText()+'-'+str(idx).zfill(5),fontcolor='blue')
        
        gl.set_value('usr_gesture', False)

    # 图像刷新速率 单位ms
    QtCore.QTimer.singleShot(1, update_figure)

def printlog(string,fontcolor):
    logtxt.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    logtxt.append("<font color="+fontcolor+">"+str(gettime)+"-->"+string+"</font>")

def print_pointcloud_log(string, fontcolor='black'):
    """输出点云采集日志到点云日志文本框"""
    global pointcloud_log_text
    if pointcloud_log_text is not None:
        pointcloud_log_text.moveCursor(QtGui.QTextCursor.End)
        gettime = time.strftime("%H:%M:%S", time.localtime())
        pointcloud_log_text.append("<font color="+fontcolor+">"+str(gettime)+" --> "+string+"</font>")
        # 限制日志行数，避免内存占用过大
        if pointcloud_log_text.document().blockCount() > 500:
            cursor = pointcloud_log_text.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.movePosition(QtGui.QTextCursor.Down, QtGui.QTextCursor.MoveAnchor, 100)
            cursor.movePosition(QtGui.QTextCursor.StartOfLine)
            cursor.movePosition(QtGui.QTextCursor.End, QtGui.QTextCursor.KeepAnchor)
            cursor.removeSelectedText()

def getradarparameters():
    if radarparameters.currentIndex() > -1 and radarparameters.currentText() != '--select--':
        radarparameters.setToolTip(radarparameters.currentText())
        configParameters = readpoint.IWR6843AOP_TLV()._initialize(config_file = radarparameters.currentText())
        rangeResolutionlabel.setText(str(configParameters["rangeResolutionMeters"])+'cm')
        dopplerResolutionlabel.setText(str(configParameters["dopplerResolutionMps"])+'m/s')
        maxRangelabel.setText(str(configParameters["maxRange"])+'m')
        maxVelocitylabel.setText(str(configParameters["maxVelocity"])+'m/s')

def openradar(config,com,data_port=None):
    global radar_ctrl
    radar_ctrl = SerialConfig(name='ConnectRadar', CLIPort=com, BaudRate=115200)
    radar_ctrl.StopRadar()
    radar_ctrl.SendConfig(config)
    
    # 启动雷达，开始发送数据
    radar_ctrl.StartRadar()
    printlog('雷达已启动，开始采集数据', fontcolor='blue')
    
    processor.start()
    print("启动数据处理线程，获取雷达信号处理结果...")
    print("当前线程：", threading.enumerate())
    processor.join(timeout=1)
    update_figure()

def updatacomstatus(cbox):
    port_list = list(list_ports.comports())
    cbox.clear()
    for i in range(len(port_list)):
        cbox.addItem(str(port_list[i][0]))

def setserialport(cbox, com):
    global CLIport_name
    global Dataport_name
    if cbox.currentIndex() > -1:
        port = cbox.currentText()
        if com == "CLI":
            CLIport_name = port
   
        else:
            Dataport_name = port
    
def sendconfigfunc():
    global CLIport_name
    global Dataport_name
    if len(CLIport_name) != 0  and radarparameters.currentText() != '--select--':
        openradar(radarparameters.currentText(), CLIport_name, Dataport_name)
        printlog(string = '发送成功', fontcolor='blue')
    else:
        printlog(string = '发送失败', fontcolor='red')


def setintervaltime():
    gl.set_value('timer_2s', True)
    QtCore.QTimer.singleShot(2000, setintervaltime)

# cnt 用来计数 200ms*cnt，代表显示多长时间
cnt = 0
def setdisplaygestureicontime():
    global _flagdisplay, cnt
    if _flagdisplay==True:
        cnt = cnt + 1
        if cnt>4:
            cnt = 0
            view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/"+str(7)+".jpg"))
            subWin.img_update("gesture_icons/"+str(7)+".jpg")
            _flagdisplay=False
    QtCore.QTimer.singleShot(200, setdisplaygestureicontime)

# 支持 Colormap切换用于图像热度色彩调整
def setcolor():
    if(color_.currentText()!='--select--' and color_.currentText()!=''):
        if color_.currentText() == 'customize':
            pgColormap = pg_get_cmap(color_.currentText())
        else:
            cmap=plt.cm.get_cmap(color_.currentText())
            pgColormap = pg_get_cmap(cmap)
        lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
        img_rdi.setLookupTable(lookup_table)
        img_rai.setLookupTable(lookup_table)
        img_rti.setLookupTable(lookup_table)
        img_dti.setLookupTable(lookup_table)
        img_rei.setLookupTable(lookup_table)

def get_filelist(dir,Filelist):
    newDir=dir
    #注意看dir是文件名还是路径＋文件名！！！！！！！！！！！！！！
    if os.path.isfile(dir):
        dir_ = os.path.basename(dir)  
        if (dir_[:2] == 'DT') and (dir_[-4:] == '.npy'):
            Filelist[0].append(dir)
        elif (dir_[:2] == 'RT') and (dir_[-4:] == '.npy'):
            Filelist[1].append(dir)
        elif (dir_[:3] == 'RDT') and (dir_[-4:] == '.npy'):
            Filelist[2].append(dir)
        elif (dir_[:3] == 'ART') and (dir_[-4:] == '.npy'):
            Filelist[3].append(dir)    
        elif (dir_[:3] == 'ERT') and (dir_[-4:] == '.npy'):
            Filelist[4].append(dir)  
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            get_filelist(newDir,Filelist)
    return Filelist

# 将特征保存为.npy文件供后续训练使用
def savedatasetsencefile():
    global datasetsencefile,start_captureidx,idx
    datasetsencefile = datasetfile+'/'+ whodatafile.text()+'/'+datasetfilebox.currentText()
    if not os.path.exists(datasetsencefile):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(datasetsencefile)

    featurelist = get_filelist(datasetsencefile, [[] for i in range(5)])
    start_captureidx = len(featurelist[0])
    idx = start_captureidx


def show_sub():
    subWin.show()
    MainWindow.hide()



def application():
    global color_,radarparameters,maxVelocitylabel,maxRangelabel,dopplerResolutionlabel,rangeResolutionlabel,logtxt
    global Recognizebtn,CaptureDatabtn,view_gesture,modelfile,datasetfilebox,whodatafile
    global img_rdi, img_rai, img_rti, img_rei, img_dti,ui
    global subWin,MainWindow
    global pointcloud_scatter, pointcloud_view, pointcloud_info_label, Dataportbox
    global pointcloud_max_points, pointcloud_refresh_rate, pointcloud_threshold
    global pointcloud_show_grid, pointcloud_show_axes
    global pointcloud_grid, pointcloud_grid_xy, pointcloud_grid_xz, pointcloud_grid_yz
    global pointcloud_x_axis, pointcloud_y_axis, pointcloud_z_axis
    global pointcloud_x_marker, pointcloud_y_marker, pointcloud_z_marker
    global pointcloud_last_update_time, pointcloud_refresh_interval
    global pointcloud_log_text, pointcloud_coordinate_info_label
    global trajectory_plot, trajectory_history, trajectory_view, trajectory_info_label, trajectory_plot_widget
    global trajectory_history_length, trajectory_point_size, trajectory_show_axes, trajectory_show_grid
    global target_count, trajectory_line_items, trajectory_scatter_items
    global trajectory_clear_button
    global height_plot, height_history, height_view, height_info_label
    global height_history_length, height_point_size, height_show_axes, height_show_grid
    global height_clear_button, height_line_ref
    global fall_time_window, fall_height_threshold, fall_height_min, fall_height_max
    global fall_alert_label, last_fall_alert_time, fall_alert_cooldown
    global fall_detection_enabled, fall_new_strategy_enabled, fall_new_strategy_sensitivity, fall_new_strategy_enabled
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()

    ui.setupUi(MainWindow)
    subWin = Qt_pet(MainWindow)
    
    # 改了D:\Applications\anaconda3\Lib\site-packages\pyqtgraph\graphicsItems\ViewBox
    # 里的ViewBox.py第919行padding = self.suggestPadding(ax)改成padding = 0

    # 距离-多普勒图
    view_rdi = ui.graphicsView_6.addViewBox()
    ui.graphicsView_6.setCentralWidget(view_rdi)#去边界
    # 距离-方位角图
    view_rai = ui.graphicsView_4.addViewBox()
    ui.graphicsView_4.setCentralWidget(view_rai)#去边界
    # 时间-距离图
    # 绘制图像（2）：在ViewBox中创建一个画布view_rti
    view_rti = ui.graphicsView.addViewBox()
    ui.graphicsView.setCentralWidget(view_rti)#去边界
    # 多普勒-时间图
    view_dti = ui.graphicsView_2.addViewBox()
    ui.graphicsView_2.setCentralWidget(view_dti)#去边界
    # 距离-俯仰角图
    view_rei = ui.graphicsView_3.addViewBox()
    ui.graphicsView_3.setCentralWidget(view_rei)#去边界
    # 手势输出图
    view_gesture = ui.graphicsView_5
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))
    view_gesture.setAlignment(QtCore.Qt.AlignCenter)


    sendcfgbtn = ui.pushButton_11
    exitbtn = ui.pushButton_12
    Recognizebtn = ui.pushButton_15
    CaptureDatabtn = ui.pushButton

    color_ = ui.comboBox
    modelfile = ui.comboBox_2
    datasetfilebox = ui.comboBox_3
    radarparameters = ui.comboBox_7
    Cliportbox = ui.comboBox_8
    Dataportbox = ui.comboBox_10

    logtxt = ui.textEdit
    whodatafile = ui.lineEdit_6
    changepage = ui.actionload
    

    rangeResolutionlabel = ui.label_14
    dopplerResolutionlabel = ui.label_35
    maxRangelabel = ui.label_16
    maxVelocitylabel = ui.label_37
    
    # 点云显示相关
    pointcloud_view = ui.graphicsView_pointcloud
    pointcloud_info_label = ui.label_pointcloud_info
    pointcloud_log_text = ui.textEdit_pointcloud_log
    pointcloud_coordinate_info_label = ui.label_coordinate_info
    
    # 点云配置控件
    pointcloud_max_points = ui.spinBox_max_points
    pointcloud_refresh_rate = ui.spinBox_refresh_rate
    pointcloud_threshold = ui.doubleSpinBox_threshold
    pointcloud_show_grid = ui.checkBox_show_grid
    pointcloud_show_axes = ui.checkBox_show_axes
    pointcloud_test_button = ui.pushButton_test_pointcloud
    pointcloud_enable_clustering = ui.checkBox_enable_clustering
    pointcloud_cluster_eps = ui.doubleSpinBox_cluster_eps
    pointcloud_cluster_min_samples = ui.spinBox_cluster_min_samples
    
    # 设置默认参数值（根据用户界面配置）
    pointcloud_max_points.setValue(1000)  # 最大点云数量: 1000
    pointcloud_refresh_rate.setValue(20)  # 刷新速率: 20ms
    pointcloud_threshold.setValue(0.45)  # 检测阈值比例: 0.45
    pointcloud_show_grid.setChecked(True)  # 显示网格: 已勾选
    pointcloud_show_axes.setChecked(True)  # 显示坐标系: 已勾选
    pointcloud_enable_clustering.setChecked(False)  # 显示聚类后的点云: 未勾选
    pointcloud_cluster_eps.setValue(0.15)  # 聚类距离阈值: 0.15m
    pointcloud_cluster_min_samples.setValue(5)  # 最小聚类点数: 5
    
    # 初始化阈值参数到全局变量
    gl.set_value('pointcloud_threshold', pointcloud_threshold.value())
    
    # 测试点云按钮功能
    def generate_test_pointcloud():
        """生成测试点云用于调试"""
        # 生成一些测试点：一个立方体形状的点云
        test_points = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    range_val = np.sqrt(x**2 + y**2 + z**2)
                    if range_val > 0.1:  # 排除原点
                        test_points.append([range_val, x, y, z])
        
        test_pointcloud = np.array(test_points, dtype=np.float32)
        # 清空队列并放入测试数据
        while not PointCloudData.empty():
            try:
                PointCloudData.get_nowait()
            except:
                break
        PointCloudData.put(test_pointcloud)
        printlog(f'生成测试点云: {len(test_pointcloud)} 个点', fontcolor='green')
        print_pointcloud_log(f'测试点云已生成: {len(test_pointcloud)} 个点', fontcolor='green')
    
    pointcloud_test_button.clicked.connect(generate_test_pointcloud)
    
    # 配置3D点云显示（pointcloud_view已经是GLViewWidget）
    pointcloud_view.setCameraPosition(distance=10, elevation=30, azimuth=45)
    # 改善背景色：使用深灰色而不是纯黑色，更容易看清
    pointcloud_view.setBackgroundColor('#1e1e1e')  # 深灰色背景
    
    # 坐标系原点设置为(-1, -1, -1)
    origin_x, origin_y, origin_z = -1, -1, -1
    axis_length = 10  # 坐标轴长度
    
    # 创建三个平面的网格（每个平面都有网格，使用相同的样式）
    grid_color = (120, 120, 120, 150)  # 统一的网格颜色和透明度
    grid_spacing = 1  # 网格间距（米）
    
    # XY平面网格（Z=-1平面，水平面）
    grid_xy = GLGridItem()
    grid_xy.setSize(x=axis_length, y=axis_length, z=0)  # XY平面，z=0
    grid_xy.setSpacing(x=grid_spacing, y=grid_spacing, z=grid_spacing)
    grid_xy.setColor(grid_color)
    grid_xy.translate(origin_x + axis_length/2, origin_y + axis_length/2, origin_z)  # 移动到Z=-1平面
    pointcloud_view.addItem(grid_xy)
    
    # XZ平面网格（Y=-1平面，侧视图）
    grid_xz = GLGridItem()
    grid_xz.setSize(x=axis_length, y=0, z=axis_length)  # XZ平面，y=0
    grid_xz.setSpacing(x=grid_spacing, y=grid_spacing, z=grid_spacing)
    grid_xz.setColor(grid_color)  # 使用和XY平面相同的颜色
    grid_xz.rotate(90, 1, 0, 0)  # 绕X轴旋转90度
    grid_xz.translate(origin_x + axis_length/2, origin_y, origin_z + axis_length/2)  # 移动到Y=-1平面
    pointcloud_view.addItem(grid_xz)
    
    # YZ平面网格（X=-1平面，前视图）
    grid_yz = GLGridItem()
    grid_yz.setSize(x=0, y=axis_length, z=axis_length)  # YZ平面，x=0
    grid_yz.setSpacing(x=grid_spacing, y=grid_spacing, z=grid_spacing)
    grid_yz.setColor(grid_color)  # 使用和XY平面相同的颜色
    grid_yz.rotate(90, 0, 1, 0)  # 绕Y轴旋转90度
    grid_yz.translate(origin_x, origin_y + axis_length/2, origin_z + axis_length/2)  # 移动到X=-1平面
    pointcloud_view.addItem(grid_yz)
    
    # 创建坐标系（X, Y, Z轴），从原点(-5, -5, -5)开始
    # X轴 - 红色（前方）
    x_axis = GLLinePlotItem()
    x_axis.setData(pos=np.array([[origin_x, origin_y, origin_z], 
                                 [origin_x + axis_length, origin_y, origin_z]]), 
                   color=(1, 0, 0, 1), width=5)
    pointcloud_view.addItem(x_axis)
    
    # Y轴 - 绿色（左侧）
    y_axis = GLLinePlotItem()
    y_axis.setData(pos=np.array([[origin_x, origin_y, origin_z], 
                                 [origin_x, origin_y + axis_length, origin_z]]), 
                   color=(0, 1, 0, 1), width=5)
    pointcloud_view.addItem(y_axis)
    
    # Z轴 - 蓝色（上方）
    z_axis = GLLinePlotItem()
    z_axis.setData(pos=np.array([[origin_x, origin_y, origin_z], 
                                 [origin_x, origin_y, origin_z + axis_length]]), 
                   color=(0, 0, 1, 1), width=5)
    pointcloud_view.addItem(z_axis)
    
    # 在坐标轴末端添加箭头标记
    arrow_size = 0.3
    # X轴末端标记（前方）- 红色箭头
    x_arrow_points = np.array([
        [origin_x + axis_length, origin_y, origin_z],  # 箭头尖端
        [origin_x + axis_length - arrow_size, origin_y + arrow_size*0.2, origin_z],  # 箭头左翼
        [origin_x + axis_length - arrow_size, origin_y - arrow_size*0.2, origin_z], # 箭头右翼
    ], dtype=np.float32)
    x_marker = GLScatterPlotItem()
    x_marker.setData(pos=x_arrow_points, color=np.array([(1, 0, 0, 1)] * len(x_arrow_points), dtype=np.float32), size=12)
    pointcloud_view.addItem(x_marker)
    
    # Y轴末端标记（左侧）- 绿色箭头
    y_arrow_points = np.array([
        [origin_x, origin_y + axis_length, origin_z],  # 箭头尖端
        [origin_x - arrow_size*0.2, origin_y + axis_length - arrow_size, origin_z], # 箭头左翼
        [origin_x + arrow_size*0.2, origin_y + axis_length - arrow_size, origin_z],  # 箭头右翼
    ], dtype=np.float32)
    y_marker = GLScatterPlotItem()
    y_marker.setData(pos=y_arrow_points, color=np.array([(0, 1, 0, 1)] * len(y_arrow_points), dtype=np.float32), size=12)
    pointcloud_view.addItem(y_marker)
    
    # Z轴末端标记（上方）- 蓝色箭头
    z_arrow_points = np.array([
        [origin_x, origin_y, origin_z + axis_length],  # 箭头尖端
        [origin_x + arrow_size*0.2, origin_y, origin_z + axis_length - arrow_size],  # 箭头左翼
        [origin_x - arrow_size*0.2, origin_y, origin_z + axis_length - arrow_size], # 箭头右翼
    ], dtype=np.float32)
    z_marker = GLScatterPlotItem()
    z_marker.setData(pos=z_arrow_points, color=np.array([(0, 0, 1, 1)] * len(z_arrow_points), dtype=np.float32), size=12)
    pointcloud_view.addItem(z_marker)
    
    # 保存坐标系和网格引用，以便控制显示/隐藏
    pointcloud_grid = grid_xy  # 主网格（XY平面）
    pointcloud_grid_xy = grid_xy  # XY平面网格
    pointcloud_grid_xz = grid_xz  # XZ平面网格
    pointcloud_grid_yz = grid_yz  # YZ平面网格
    pointcloud_x_axis = x_axis
    pointcloud_y_axis = y_axis
    pointcloud_z_axis = z_axis
    pointcloud_x_marker = x_marker
    pointcloud_y_marker = y_marker
    pointcloud_z_marker = z_marker
    
    # 创建散点图项
    pointcloud_scatter = GLScatterPlotItem()
    pointcloud_scatter.setGLOptions('opaque')
    pointcloud_view.addItem(pointcloud_scatter)
    
    # 点云刷新速率控制变量
    pointcloud_last_update_time = 0
    pointcloud_refresh_interval = 50  # 默认50ms
    
    # 轨迹显示相关
    trajectory_view_container = ui.graphicsView_trajectory
    trajectory_info_label = ui.label_trajectory_info
    trajectory_history_length = ui.spinBox_trajectory_history
    trajectory_point_size = ui.spinBox_trajectory_point_size
    trajectory_show_axes = ui.checkBox_trajectory_show_axes
    trajectory_show_grid = ui.checkBox_trajectory_show_grid
    target_count = ui.spinBox_target_count
    trajectory_clear_button = ui.pushButton_clear_trajectory
    
    # 初始化轨迹历史存储（存储格式: [(timestamp, target_id, x, y, range), ...]）
    trajectory_history = []
    
    # 存储轨迹连线对象（用于多目标显示）
    global trajectory_line_items, trajectory_scatter_items
    trajectory_line_items = []
    trajectory_scatter_items = []
    
    # 雷达参数：距离分辨率0.09m，最大距离5.4m
    MAX_RANGE = 5.4  # 最大距离5.4米
    RANGE_RESOLUTION = 0.09  # 距离分辨率0.09米
    
    # 配置轨迹2D显示（XY平面，类似高度检测页面的显示方式）
    trajectory_plot_widget = trajectory_view_container.addPlot(title=f"目标运动轨迹 (XY平面) | 距离分辨率: {RANGE_RESOLUTION*100:.1f}cm | 范围: ±{MAX_RANGE:.1f}m")
    trajectory_plot_widget.setLabel('left', 'Y (m)', color='black', size='12pt')  # Y轴（垂直）
    trajectory_plot_widget.setLabel('bottom', 'X (m)', color='black', size='12pt')  # X轴（水平）
    trajectory_plot_widget.showGrid(x=True, y=True, alpha=0.3)
    trajectory_plot_widget.setAspectLocked(True)  # 锁定纵横比
    # 设置显示范围
    trajectory_plot_widget.setXRange(-MAX_RANGE, MAX_RANGE)  # X轴范围
    trajectory_plot_widget.setYRange(-MAX_RANGE, MAX_RANGE)  # Y轴范围
    
    # 绘制坐标轴（X轴水平，Y轴垂直）
    # X轴（红色，水平）
    x_axis_line_traj = pg.PlotDataItem([-MAX_RANGE, MAX_RANGE], [0, 0], pen=pg.mkPen(color='r', width=2), name='X轴')
    trajectory_plot_widget.addItem(x_axis_line_traj)
    # Y轴（绿色，垂直）
    y_axis_line_traj = pg.PlotDataItem([0, 0], [-MAX_RANGE, MAX_RANGE], pen=pg.mkPen(color='g', width=2), name='Y轴')
    trajectory_plot_widget.addItem(y_axis_line_traj)
    
    # 标注原点
    origin_text_traj = pg.TextItem('雷达原点 (0,0)', anchor=(0.5, 1.5), color='blue')
    origin_text_traj.setPos(0, 0)
    # 设置字体大小
    font_traj = QtGui.QFont()
    font_traj.setPointSize(10)
    origin_text_traj.setFont(font_traj)
    trajectory_plot_widget.addItem(origin_text_traj)
    
    # 标注X轴方向（水平，右侧）
    x_axis_text_traj = pg.TextItem('X轴 (前方)', anchor=(0.5, 0.5), color='red')
    x_axis_text_traj.setPos(MAX_RANGE * 0.7, 0.2)  # 放在右侧
    x_axis_text_traj.setFont(font_traj)
    trajectory_plot_widget.addItem(x_axis_text_traj)
    
    # 标注Y轴方向（垂直，上方）
    y_axis_text_traj = pg.TextItem('Y轴 (右侧)', anchor=(0.5, 0.5), color='green')
    y_axis_text_traj.setPos(0.2, MAX_RANGE * 0.7)  # 放在上方
    y_axis_text_traj.setFont(font_traj)
    trajectory_plot_widget.addItem(y_axis_text_traj)
    
    # 轨迹散点图（2D）
    trajectory_plot = pg.ScatterPlotItem(size=trajectory_point_size.value(), pen=pg.mkPen(width=1), brush=pg.mkBrush(255, 0, 0, 200))
    trajectory_plot_widget.addItem(trajectory_plot)
    
    # 轨迹连线（可选，用于显示轨迹路径）
    trajectory_line_ref = pg.PlotDataItem(pen=pg.mkPen(color='blue', width=2, style=QtCore.Qt.DashLine), name='轨迹连线')
    trajectory_plot_widget.addItem(trajectory_line_ref)
    
    # 保留trajectory_view_container引用，用于兼容性
    trajectory_view = trajectory_view_container
    
    # 清除轨迹按钮功能
    def clear_trajectory():
        global trajectory_history, trajectory_line_items, trajectory_scatter_items
        trajectory_history = []
        trajectory_plot.setData([], [])
        trajectory_line_ref.setData([], [])
        # 清除所有连线和散点图
        for line_item in trajectory_line_items:
            trajectory_plot_widget.removeItem(line_item)
        for scatter_item in trajectory_scatter_items:
            trajectory_plot_widget.removeItem(scatter_item)
        trajectory_line_items = []
        trajectory_scatter_items = []
        trajectory_info_label.setText("轨迹信息: 已清除")
    
    trajectory_clear_button.clicked.connect(clear_trajectory)
    
    # 控制网格和坐标轴的显示
    def update_trajectory_display():
        if trajectory_show_grid.isChecked():
            trajectory_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        else:
            trajectory_plot_widget.showGrid(x=False, y=False)
        
        if trajectory_show_axes.isChecked():
            x_axis_line_traj.show()
            y_axis_line_traj.show()
            origin_text_traj.show()
            x_axis_text_traj.show()
            y_axis_text_traj.show()
        else:
            x_axis_line_traj.hide()
            y_axis_line_traj.hide()
            origin_text_traj.hide()
            x_axis_text_traj.hide()
            y_axis_text_traj.hide()
    
    trajectory_show_grid.stateChanged.connect(lambda: update_trajectory_display())
    trajectory_show_axes.stateChanged.connect(lambda: update_trajectory_display())
    
    # 高度检测显示相关
    height_view = ui.graphicsView_height
    height_info_label = ui.label_height_info
    height_history_length = ui.spinBox_height_history
    height_point_size = ui.spinBox_height_point_size
    height_show_axes = ui.checkBox_height_show_axes
    height_show_grid = ui.checkBox_height_show_grid
    height_clear_button = ui.pushButton_clear_height
    fall_time_window = ui.doubleSpinBox_fall_time_window
    fall_height_threshold = ui.doubleSpinBox_fall_height_threshold
    fall_height_min = ui.doubleSpinBox_fall_height_min
    fall_height_max = ui.doubleSpinBox_fall_height_max
    fall_detection_enabled = ui.checkBox_enable_fall_detection  # 跌倒检测开关
    fall_new_strategy_enabled = ui.checkBox_enable_new_fall_strategy  # 新策略跌倒检测开关
    fall_new_strategy_sensitivity = ui.comboBox_new_strategy_sensitivity  # 新策略灵敏度等级
    fall_alert_label = ui.label_fall_alert  # 跌倒提示标签
    
    # 初始化高度历史存储（存储格式: [(timestamp, z, y, range), ...]）
    height_history = []
    
    # 初始化跌倒检测相关变量
    last_fall_alert_time = 0  # 上次跌倒提示时间，避免重复提示
    fall_alert_cooldown = 3000  # 跌倒提示冷却时间（毫秒），3秒内不重复提示
    
    # 跌倒检测开关状态改变时的处理函数
    def on_fall_detection_toggled(enabled):
        """当跌倒检测开关状态改变时调用"""
        if not enabled:
            # 如果关闭跌倒检测，隐藏跌倒提示标签
            if fall_alert_label is not None:
                fall_alert_label.hide()
            # 禁用所有跌倒检测相关控件
            fall_new_strategy_enabled.setEnabled(False)
            fall_new_strategy_sensitivity.setEnabled(False)
            fall_time_window.setEnabled(False)
            fall_height_threshold.setEnabled(False)
            fall_height_min.setEnabled(False)
            fall_height_max.setEnabled(False)
            # 禁用标签（通过设置样式使其变灰）
            ui.label_new_strategy_sensitivity.setEnabled(False)
            ui.label_fall_time_window.setEnabled(False)
            ui.label_fall_height_threshold.setEnabled(False)
            ui.label_fall_height_min.setEnabled(False)
            ui.label_fall_height_max.setEnabled(False)
        else:
            # 启用新策略选项
            fall_new_strategy_enabled.setEnabled(True)
            # 根据新策略状态决定是否启用旧策略参数
            on_new_strategy_toggled(fall_new_strategy_enabled.isChecked())
    
    # 获取新策略灵敏度参数
    # 新策略开关状态改变时的处理函数
    def on_new_strategy_toggled(enabled):
        """当新策略开关状态改变时调用"""
        if enabled:
            # 启用新策略时，禁用旧策略的参数，启用灵敏度选择
            fall_time_window.setEnabled(False)
            fall_height_threshold.setEnabled(False)
            fall_height_min.setEnabled(False)
            fall_height_max.setEnabled(False)
            # 禁用标签
            ui.label_fall_time_window.setEnabled(False)
            ui.label_fall_height_threshold.setEnabled(False)
            ui.label_fall_height_min.setEnabled(False)
            ui.label_fall_height_max.setEnabled(False)
            # 启用灵敏度选择
            fall_new_strategy_sensitivity.setEnabled(True)
            ui.label_new_strategy_sensitivity.setEnabled(True)
        else:
            # 禁用新策略时，启用旧策略的参数（前提是跌倒检测已启用），禁用灵敏度选择
            if fall_detection_enabled.isChecked():
                fall_time_window.setEnabled(True)
                fall_height_threshold.setEnabled(True)
                fall_height_min.setEnabled(True)
                fall_height_max.setEnabled(True)
                # 启用标签
                ui.label_fall_time_window.setEnabled(True)
                ui.label_fall_height_threshold.setEnabled(True)
                ui.label_fall_height_min.setEnabled(True)
                ui.label_fall_height_max.setEnabled(True)
            # 禁用灵敏度选择
            fall_new_strategy_sensitivity.setEnabled(False)
            ui.label_new_strategy_sensitivity.setEnabled(False)
    
    # 连接跌倒检测开关的信号
    fall_detection_enabled.stateChanged.connect(on_fall_detection_toggled)
    # 连接新策略开关的信号
    fall_new_strategy_enabled.stateChanged.connect(on_new_strategy_toggled)
    
    # 初始化控件状态（根据默认值设置）
    on_fall_detection_toggled(fall_detection_enabled.isChecked())
    
    # 配置高度2D显示（Z-Y轴，Z轴朝上）
    # 高度范围：-1m到1m
    HEIGHT_MAX_RANGE = 1.0  # 高度最大范围1米
    height_plot_widget = height_view.addPlot(title=f"高度检测 (ZY平面) | 距离分辨率: {RANGE_RESOLUTION*100:.1f}cm | 高度范围: ±{HEIGHT_MAX_RANGE:.1f}m")
    height_plot_widget.setLabel('left', 'Z (m)', color='black', size='12pt')  # Z轴朝上（垂直）
    height_plot_widget.setLabel('bottom', 'Y (m)', color='black', size='12pt')  # Y轴水平
    height_plot_widget.showGrid(x=True, y=True, alpha=0.3)
    height_plot_widget.setAspectLocked(True)  # 锁定纵横比
    # 设置显示范围（高度范围-1m到1m，Y轴保持原范围）
    height_plot_widget.setXRange(-MAX_RANGE, MAX_RANGE)  # X轴显示Y值（水平，保持原范围）
    height_plot_widget.setYRange(-HEIGHT_MAX_RANGE, HEIGHT_MAX_RANGE)  # Y轴显示Z值（垂直，朝上，-1m到1m）
    
    # 绘制坐标轴（Z轴朝上，Y轴水平）
    # Y轴（绿色，水平）- X轴位置显示Y值
    y_axis_line_height = pg.PlotDataItem([-MAX_RANGE, MAX_RANGE], [0, 0], pen=pg.mkPen(color='g', width=2), name='Y轴')
    height_plot_widget.addItem(y_axis_line_height)
    # Z轴（蓝色，垂直朝上）- Y轴位置显示Z值，范围-1m到1m
    z_axis_line = pg.PlotDataItem([0, 0], [-HEIGHT_MAX_RANGE, HEIGHT_MAX_RANGE], pen=pg.mkPen(color='b', width=2), name='Z轴')
    height_plot_widget.addItem(z_axis_line)
    
    # 标注原点
    origin_text_height = pg.TextItem('雷达原点 (0,0)', anchor=(0.5, 1.5), color='blue')
    origin_text_height.setPos(0, 0)
    # 设置字体大小
    font_height = QtGui.QFont()
    font_height.setPointSize(10)
    origin_text_height.setFont(font_height)
    height_plot_widget.addItem(origin_text_height)
    
    # 标注Y轴方向（水平，左侧）
    y_axis_text_height = pg.TextItem('Y轴 (左侧)', anchor=(0.5, 0.5), color='green')
    y_axis_text_height.setPos(-MAX_RANGE * 0.7, 0.2)  # 放在左侧
    y_axis_text_height.setFont(font_height)
    height_plot_widget.addItem(y_axis_text_height)
    
    # 标注Z轴方向（垂直，上方）
    z_axis_text = pg.TextItem('Z轴 (上方)', anchor=(0.5, 0.5), color='blue')
    z_axis_text.setPos(0.2, HEIGHT_MAX_RANGE * 0.7)  # 放在上方，使用HEIGHT_MAX_RANGE
    z_axis_text.setFont(font_height)
    height_plot_widget.addItem(z_axis_text)
    
    # 高度散点图
    height_plot = pg.ScatterPlotItem(size=height_point_size.value(), pen=pg.mkPen(width=1), brush=pg.mkBrush(255, 0, 0, 200))
    height_plot_widget.addItem(height_plot)
    
    # 高度连线
    height_line_ref = pg.PlotDataItem(pen=pg.mkPen(color='blue', width=2, style=QtCore.Qt.DashLine), name='高度轨迹')
    height_plot_widget.addItem(height_line_ref)
    
    # 控制网格和坐标轴的显示
    def update_height_display():
        if height_show_grid.isChecked():
            height_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        else:
            height_plot_widget.showGrid(x=False, y=False)
        
        if height_show_axes.isChecked():
            z_axis_line.show()
            y_axis_line_height.show()
            origin_text_height.show()
            z_axis_text.show()
            y_axis_text_height.show()
        else:
            z_axis_line.hide()
            y_axis_line_height.hide()
            origin_text_height.hide()
            z_axis_text.hide()
            y_axis_text_height.hide()
    
    height_show_grid.stateChanged.connect(lambda: update_height_display())
    height_show_axes.stateChanged.connect(lambda: update_height_display())
    
    # 清除高度按钮功能
    def clear_height():
        global height_history
        height_history = []
        height_plot.setData([], [])
        height_line_ref.setData([], [])
        height_info_label.setText("高度信息: 已清除")
    
    height_clear_button.clicked.connect(clear_height)

    # ---------------------------------------------------
    # lock the aspect ratio so pixels are always square
    # view_rai.setAspectLocked(True)
    # view_rti.setAspectLocked(True)
    img_rdi = pg.ImageItem(border=None)
    img_rai = pg.ImageItem(border=None)
    # 时间-距离图
    # 绘制图像（1）：通过pyqtgraph创建图像容器img_rti，内部封装了矩阵图像的显示和着色逻辑
    img_rti = pg.ImageItem(border=None)
    img_dti = pg.ImageItem(border=None)
    img_rei = pg.ImageItem(border=None)

    # Colormap
    pgColormap = pg_get_cmap('customize')
    lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
    img_rdi.setLookupTable(lookup_table)
    img_rai.setLookupTable(lookup_table)
    img_rti.setLookupTable(lookup_table)
    img_dti.setLookupTable(lookup_table)
    img_rei.setLookupTable(lookup_table)

    view_rdi.addItem(img_rdi)
    view_rai.addItem(img_rai)
    # 绘制图像（2）：将数据容器img_rti添加到画布view_rti中
    # 至此，img_rti已经“放置”到了GUI界面的时间-距离图窗口view_rti中，但还没绘图内容（img_rti容器为空）
    # 随后，后台线程processor负责从UDP监听队列中读取原始数据，解析后塞入RTIData队列中
    view_rti.addItem(img_rti)
    view_dti.addItem(img_dti)
    view_rei.addItem(img_rei)
    

    Cliportbox.arrowClicked.connect(lambda:updatacomstatus(Cliportbox)) 
    Cliportbox.currentIndexChanged.connect(lambda:setserialport(Cliportbox, com = 'CLI'))
    Dataportbox.arrowClicked.connect(lambda:updatacomstatus(Dataportbox))
    Dataportbox.currentIndexChanged.connect(lambda:setserialport(Dataportbox, com = 'Data'))
    color_.currentIndexChanged.connect(setcolor)
    modelfile.currentIndexChanged.connect(loadmodel)
    radarparameters.currentIndexChanged.connect(getradarparameters)
    datasetfilebox.currentIndexChanged.connect(savedatasetsencefile)
    whodatafile.editingFinished.connect(savedatasetsencefile)
    # send按键 信号-槽函数
    sendcfgbtn.clicked.connect(sendconfigfunc)
    Recognizebtn.clicked.connect(setintervaltime)
    # Recognizebtn.clicked.connect(setdisplaygestureicontime)
    CaptureDatabtn.clicked.connect(setintervaltime)
    changepage.triggered.connect(show_sub)
    # 2022/2/24 添加小型化控件 不能正常退出了
    exitbtn.clicked.connect(app.instance().exit)
    
    # 点云配置控件信号连接（这些控件值的变化会自动在update_figure中生效）
    # 不需要额外连接，因为update_figure中会读取这些控件的值



    app.instance().exec_()


    try:
        if radar_ctrl.CLIPort:
            if radar_ctrl.CLIPort.isOpen():
                radar_ctrl.StopRadar()
    except:
        pass























if __name__ == '__main__':
    print("进程启动...")
    # Queue for access data
    BinData = Queue() # 原始数据队列

    # 时间信息

    # ZHChen use 2025-05-20 ---
    RTIData = Queue() # 时间距离图队列
    # ZHChen use 2025-05-20 ---
    DTIData = Queue() # 多普勒时间队列

    # 连续过程信息
    RDIData = Queue() # 距离多普勒队列
    RAIData = Queue() # 距离方位角队列
    REIData = Queue() # 方位角俯仰角队列
    
    # 点云数据队列
    PointCloudData = Queue() # 点云数据队列
    
    # 点云历史缓冲区（用于时间衰减显示）
    # 存储格式: [(timestamp_ms, pointcloud), ...]
    # 其中 pointcloud 格式: [num_points, 4] -> [range, x, y, z]
    PointCloudHistory = []  # 点云历史缓冲区
    print("创建数据队列...")


    # Radar config parameters
    NUM_TX = 3
    NUM_RX = 4
    NUM_CHIRPS = 64
    NUM_ADC_SAMPLES = 64

    # 雷达配置信息计算
    radar_config = [NUM_ADC_SAMPLES, NUM_CHIRPS, NUM_TX, NUM_RX]
    frame_length = NUM_ADC_SAMPLES * NUM_CHIRPS * NUM_TX * NUM_RX * 2
    print("计算雷达参数信息，计算帧长...")

    # config DCA1000 to receive bin data
    # dca1000_cfg是类实例化的对象
    dca1000_cfg = DCA1000Config('DCA1000Config',config_address = ('192.168.33.30', 4096),
                                                FPGA_address_cfg=('192.168.33.180', 4096))
    print("配置DCA1000地址及端口参数...")


    # 配置并启动 UDP监听线程接收雷达原始数据，即未经任何处理的雷达信号（raw.bin），collector是一个线程
    # UdpListener创建了监听线程用于接收上位机命令，并将监听线程对象返回给了collector
    collector = UdpListener('Listener', BinData, frame_length)
    print("创建UDP监听线程，用于获取雷达原始数据...")

    # 创建数据处理线程,将原始二进制数据转换为图像用数据结构
    global processor
    processor = DataProcessor('Processor', radar_config, BinData, RTIData, DTIData,
                                             RDIData, RAIData, REIData, PointCloudData)
    print("创建数据处理线程，用于获取雷达信号处理结果...")

    # 启动监听线程
    # 本质上是python多线程编程，整个main.py是一个进程，这里相当于是采用单进程内的多线程并发编程，多线程编程有利于异步执行
    # 例如本项目中，后台线程处理耗时操作（数据采集），不影响主线程响应（GUI更新）
    collector.start()
    print("启动UDP监听线程，开始获取雷达原始数据...")

    # 启动 PyQt5 GUI主界面
    application()

    # 关闭GUI之后发生如下：
    # 当用户关闭GUI后，关闭与dca1000的UDP连接，停止接收雷达数据
    dca1000_cfg.DCA1000_close()

    # 线程同步，超时等待1s
    # 若collector线程在1s内未结束，当前线程会继续执行后续代码（不再阻塞）
    collector.join(timeout=1)
    print("UDP监听线程同步中...")


    # 总结一下就是：
    # （1）start()启动新线程，异步执行 run() 方法
    # （2）join()阻塞当前线程，等待子线程执行完毕（常用于确保数据完整性）
    # （3）join(timeout=1)限时等待，超时后当前线程继续执行


    print("end---------程序结束---------end")
    sys.exit()