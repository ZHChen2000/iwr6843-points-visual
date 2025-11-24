"""
点云聚类模块
使用DBSCAN算法对点云进行聚类，合并距离近、强度接近的点，并过滤噪声点
"""

# ========== 必须在所有导入之前设置环境变量 ==========
import os
import multiprocessing
import sys
# 修复 Windows 上 joblib/loky 检测物理 CPU 核心数失败的问题
# 设置 LOKY_MAX_CPU_COUNT 为逻辑核心数，避免尝试检测物理核心数
os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
# 注意：Windows 上 multiprocessing 不支持 'threading' 上下文，移除 JOBLIB_START_METHOD 设置
# joblib 会自动使用合适的后端（loky 或 spawn）
# ====================================================

import numpy as np
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

# 在导入 sklearn 之前，先导入并 patch joblib，防止在导入 sklearn 时触发检测
# 这是最关键的步骤：必须在 sklearn 导入之前 patch joblib
try:
    # 在 Windows 上，确保 multiprocessing 使用正确的上下文
    # 在导入 joblib 之前设置，避免上下文错误
    if sys.platform == 'win32':
        # Windows 上使用 'spawn' 上下文（默认）
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置过，忽略错误
            pass
        
        # Patch multiprocessing 模块，拦截对 'threading' 上下文的请求
        # 因为 joblib 在导入时可能会尝试使用 'threading' 上下文，但 Windows 不支持
        try:
            # 方法1: Patch multiprocessing.get_context 函数
            _original_mp_get_context = multiprocessing.get_context
            
            def _patched_mp_get_context(method=None):
                """Patch multiprocessing.get_context，将 'threading' 请求重定向到 'spawn'"""
                if method == 'threading':
                    # Windows 不支持 threading，使用 spawn 代替
                    method = 'spawn'
                return _original_mp_get_context(method=method)
            
            multiprocessing.get_context = _patched_mp_get_context
            
            # 方法2: Patch context 模块中的 BaseContext 类
            from multiprocessing import context as mp_context
            
            # Patch BaseContext.get_context 方法
            if hasattr(mp_context, 'BaseContext'):
                _original_base_get_context = mp_context.BaseContext.get_context
                
                def _patched_base_get_context(self, method=None):
                    """Patch BaseContext.get_context，将 'threading' 请求重定向到 'spawn'"""
                    if method == 'threading':
                        method = 'spawn'
                    return _original_base_get_context(self, method=method)
                
                mp_context.BaseContext.get_context = _patched_base_get_context
            
            # 方法3: 如果存在 DefaultContext，也 patch 它
            if hasattr(mp_context, 'DefaultContext'):
                _original_default_get_context = mp_context.DefaultContext.get_context
                
                def _patched_default_get_context(self, method=None):
                    """Patch DefaultContext.get_context，将 'threading' 请求重定向到 'spawn'"""
                    if method == 'threading':
                        method = 'spawn'
                    return _original_default_get_context(self, method=method)
                
                mp_context.DefaultContext.get_context = _patched_default_get_context
                
        except (AttributeError, ImportError) as e:
            # 如果无法 patch，继续（不影响主流程）
            pass
    
    # 先导入 joblib，这样我们可以立即 patch
    import joblib
    # 尝试导入并 patch joblib.externals.loky.backend.context 模块
    try:
        from joblib.externals.loky.backend import context as loky_context
        if hasattr(loky_context, '_count_physical_cores'):
            def _patched_count_physical_cores():
                """Patch 后的函数，直接返回逻辑核心数，避免调用 wmic"""
                return multiprocessing.cpu_count()
            loky_context._count_physical_cores = _patched_count_physical_cores
    except (ImportError, AttributeError):
        # 如果 loky 模块不存在或无法 patch，尝试直接 patch subprocess 调用
        # 通过 monkey patch subprocess.run 来拦截 wmic 调用
        import subprocess
        _original_subprocess_run = subprocess.run
        
        def _patched_subprocess_run(*args, **kwargs):
            """Patch subprocess.run 来拦截 wmic CPU 检测调用"""
            # 检查是否是 wmic CPU 检测命令
            if args and len(args) > 0:
                cmd = args[0]
                cmd_str = str(cmd).lower() if isinstance(cmd, (list, tuple, str)) else ''
                # 检测 wmic 相关的 CPU 查询命令
                if 'wmic' in cmd_str and ('cpu' in cmd_str or 'numberofcores' in cmd_str or 'numberoflogicalprocessors' in cmd_str):
                    # 这是一个 CPU 检测命令，创建一个假的 CompletedProcess 对象
                    from subprocess import CompletedProcess
                    # wmic 通常返回数字，我们返回逻辑核心数
                    cpu_count = multiprocessing.cpu_count()
                    return CompletedProcess(
                        args=cmd,
                        returncode=0,
                        stdout=str(cpu_count).encode('utf-8'),
                        stderr=b''
                    )
            # 其他命令正常执行
            return _original_subprocess_run(*args, **kwargs)
        
        # 应用 patch
        subprocess.run = _patched_subprocess_run
except ImportError:
    # 如果 joblib 不存在，继续
    pass

# 导入 sklearn（此时环境变量已设置，joblib 也已 patch）
from sklearn.cluster import DBSCAN

# 在导入后再次 patch，确保后续调用时不会触发检测
try:
    from joblib.externals.loky.backend import context as loky_context
    if hasattr(loky_context, '_count_physical_cores'):
        def _patched_count_physical_cores():
            """Patch 后的函数，直接返回逻辑核心数，避免调用 wmic"""
            return multiprocessing.cpu_count()
        loky_context._count_physical_cores = _patched_count_physical_cores
except (ImportError, AttributeError):
    # 如果模块不存在或没有该函数，忽略
    pass


def cluster_pointcloud(pointcloud, eps=0.3, min_samples=3, intensity_weight=0.1):
    """
    对点云进行聚类处理
    
    Args:
        pointcloud: numpy数组，形状 (num_points, 4)，每行为 [range, x, y, z]
        eps: DBSCAN的邻域半径（米），用于判断两个点是否属于同一聚类
        min_samples: 形成聚类所需的最小点数
        intensity_weight: 强度权重（0-1），用于在距离计算中加入强度信息
                        强度用range的倒数表示（距离越近强度越高）
    
    Returns:
        clustered_pointcloud: 聚类后的点云，形状 (num_clustered_points, 4)
                             格式: [range, x, y, z]
        cluster_labels: 每个点所属的聚类标签（-1表示噪声点）
        cluster_info: 字典，包含聚类统计信息
    """
    if pointcloud is None or len(pointcloud) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([]), {
            'num_clusters': 0,
            'num_noise': 0,
            'num_points_before': 0,
            'num_points_after': 0
        }
    
    # 提取数据
    ranges = pointcloud[:, 0]  # 距离
    x = pointcloud[:, 1]       # X坐标
    y = pointcloud[:, 2]       # Y坐标
    z = pointcloud[:, 3]       # Z坐标
    
    # 计算强度（使用距离的倒数，距离越近强度越高）
    # 为了避免除零，添加小的偏移量
    intensities = 1.0 / (ranges + 0.01)
    
    # 归一化强度到0-1范围
    if intensities.max() > intensities.min():
        intensities_normalized = (intensities - intensities.min()) / (intensities.max() - intensities.min())
    else:
        intensities_normalized = np.zeros_like(intensities)
    
    # 准备特征矩阵：空间坐标 + 强度信息
    # 空间坐标使用原始值（米）
    # 强度信息需要缩放到合适的尺度，使其对距离计算有影响但不占主导
    # 假设空间坐标范围在0-10米，强度范围在0-1，需要将强度缩放到米级别
    # 使用intensity_weight来控制强度的影响程度
    spatial_coords = np.column_stack([x, y, z])
    intensity_scaled = intensities_normalized.reshape(-1, 1) * intensity_weight * eps
    
    # 组合特征：空间坐标 + 缩放的强度
    features = np.hstack([spatial_coords, intensity_scaled])
    
    # 执行DBSCAN聚类
    # eps需要根据特征空间调整（因为加入了强度维度）
    # 如果强度权重很小，eps基本就是空间距离
    clustering = DBSCAN(eps=eps * np.sqrt(1 + intensity_weight**2), min_samples=min_samples)
    cluster_labels = clustering.fit_predict(features)
    
    # 统计信息
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)
    
    # 过滤噪声点（标签为-1的点）
    valid_mask = cluster_labels != -1
    clustered_pointcloud = pointcloud[valid_mask]
    
    # 对每个聚类内的点进行合并（使用聚类中心）
    if num_clusters > 0 and len(clustered_pointcloud) > 0:
        clustered_points_list = []
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
            
            # 获取该聚类的所有点
            cluster_mask = cluster_labels == label
            cluster_points = pointcloud[cluster_mask]
            
            # 计算聚类中心（使用加权平均，权重为强度）
            cluster_ranges = cluster_points[:, 0]
            cluster_x = cluster_points[:, 1]
            cluster_y = cluster_points[:, 2]
            cluster_z = cluster_points[:, 3]
            
            # 使用强度作为权重
            cluster_intensities = 1.0 / (cluster_ranges + 0.01)
            weights = cluster_intensities / cluster_intensities.sum()
            
            # 计算加权中心
            center_x = np.average(cluster_x, weights=weights)
            center_y = np.average(cluster_y, weights=weights)
            center_z = np.average(cluster_z, weights=weights)
            center_range = np.sqrt(center_x**2 + center_y**2 + center_z**2)
            
            # 添加聚类中心点
            clustered_points_list.append([center_range, center_x, center_y, center_z])
        
        if len(clustered_points_list) > 0:
            clustered_pointcloud = np.array(clustered_points_list, dtype=np.float32)
        else:
            clustered_pointcloud = np.zeros((0, 4), dtype=np.float32)
    else:
        clustered_pointcloud = np.zeros((0, 4), dtype=np.float32)
    
    cluster_info = {
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'num_points_before': len(pointcloud),
        'num_points_after': len(clustered_pointcloud)
    }
    
    return clustered_pointcloud, cluster_labels, cluster_info


def cluster_pointcloud_simple(pointcloud, eps=0.3, min_samples=3):
    """
    简化版聚类函数，只使用空间距离，不考虑强度
    
    Args:
        pointcloud: numpy数组，形状 (num_points, 4)，每行为 [range, x, y, z]
        eps: DBSCAN的邻域半径（米）
        min_samples: 形成聚类所需的最小点数
    
    Returns:
        clustered_pointcloud: 聚类后的点云，形状 (num_clustered_points, 4)
        cluster_labels: 每个点所属的聚类标签（-1表示噪声点）
        cluster_info: 字典，包含聚类统计信息
    """
    if pointcloud is None or len(pointcloud) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.array([]), {
            'num_clusters': 0,
            'num_noise': 0,
            'num_points_before': 0,
            'num_points_after': 0
        }
    
    # 提取空间坐标
    x = pointcloud[:, 1]
    y = pointcloud[:, 2]
    z = pointcloud[:, 3]
    spatial_coords = np.column_stack([x, y, z])
    
    # 执行DBSCAN聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(spatial_coords)
    
    # 统计信息
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)
    
    # 过滤噪声点
    valid_mask = cluster_labels != -1
    clustered_pointcloud = pointcloud[valid_mask]
    
    # 对每个聚类内的点进行合并（使用聚类中心）
    if num_clusters > 0 and len(clustered_pointcloud) > 0:
        clustered_points_list = []
        for label in unique_labels:
            if label == -1:
                continue
            
            # 获取该聚类的所有点
            cluster_mask = cluster_labels == label
            cluster_points = pointcloud[cluster_mask]
            
            # 计算聚类中心（简单平均）
            center_x = np.mean(cluster_points[:, 1])
            center_y = np.mean(cluster_points[:, 2])
            center_z = np.mean(cluster_points[:, 3])
            center_range = np.sqrt(center_x**2 + center_y**2 + center_z**2)
            
            clustered_points_list.append([center_range, center_x, center_y, center_z])
        
        if len(clustered_points_list) > 0:
            clustered_pointcloud = np.array(clustered_points_list, dtype=np.float32)
        else:
            clustered_pointcloud = np.zeros((0, 4), dtype=np.float32)
    else:
        clustered_pointcloud = np.zeros((0, 4), dtype=np.float32)
    
    cluster_info = {
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'num_points_before': len(pointcloud),
        'num_points_after': len(clustered_pointcloud)
    }
    
    return clustered_pointcloud, cluster_labels, cluster_info

