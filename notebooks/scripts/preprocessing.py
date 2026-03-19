import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def read_and_process(geo,rel,dyna,ext,fut):
    dyna['time'] = pd.to_datetime(dyna['time'])

    # 对 dyna 的time 数值时间戳，并转换为数值型时间戳（秒级）
    dyna['time_unix'] = dyna['time'].astype('int64') // 10**9

    # 按照 time 和entity_id	排序
    dyna = dyna.sort_values(by=['time','entity_id'])

    dyna.drop(columns=['dyna_id','type'], inplace=True)


    # 创建一个三维数组[时间步数, 节点数, 特征维度]
    time_steps = dyna['time'].nunique()
    nodes = dyna['entity_id'].nunique()
    features = dyna.drop(columns=['time','entity_id']).shape[1]

    # 初始化一个空的三维数组
    dyna_array = np.zeros((time_steps, nodes, features))

    # 填充数组
    for i, (time, group) in enumerate(dyna.groupby('time')):
        dyna_array[i, :, :] = group.drop(columns=['time', 'entity_id']).values

    # dyna_traffic 前两个; time 为最后一个
    dyna_traffic = dyna_array[:, :, :2]
    time_unix = dyna_array[:, :, -1]

    ext['time'] = pd.to_datetime(ext['time'])

    # 按照 time 和 geo_id 排序
    ext = ext.sort_values(by=['time','geo_id'])

    ext.drop(columns=['ext_id'], inplace=True)

    time_steps = ext['time'].nunique()
    nodes = ext['geo_id'].nunique()
    features = ext.drop(columns=['time', 'geo_id']).shape[1]

    # 初始化一个空的三维数组
    ext_array = np.zeros((time_steps, nodes, features))

    # 填充数组
    for i, (time, group) in enumerate(ext.groupby('time')):
        ext_array[i, :, :] = group.drop(columns=['time', 'geo_id']).values

    #ext_weather 前四个; ind_seq 为后三个
    ext_weather = ext_array[:, :, :4]
    ind_seq = ext_array[:, :, -3:]


    # 将时间列转换为 datetime 类型
    fut['time'] = pd.to_datetime(fut['time'])

    # time  数值时间戳，并转换为数值型时间戳（秒级）
    fut['time'] = fut['time'].astype('int64') // 10**9

    # 按照time 和 geo_id 排序
    fut = fut.sort_values(by=['time', 'geo_id'])

    # 创建一个三维数组 [时间步数, 节点数, 特征维度]
    time_steps = fut['time'].nunique()
    nodes= fut['geo_id'].nunique()
    features = fut.drop(columns=['time','geo_id','fut_id']).shape[1]

    # 初始化一个空的三维数组
    ind_hor = np.zeros((time_steps, nodes, features))

    # 填充数组
    for i, (time, group) in enumerate(fut.groupby('time')):
        ind_hor[i, :, :] = group.drop(columns=['time', 'geo_id','fut_id']).values

    # 提取 unique 的节点 ID
    nodes = sorted(set(rel['origin_id']).union(set(rel['destination_id'])))
    num_nodes = len(nodes)

    # 创建一个空的邻接矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 填充邻接矩阵
    for _, row in rel.iterrows():
        origin = row['origin_id'] - 1  # 调整为零基索引
        destination = row['destination_id'] - 1  # 调整为零基索引
        weight = row['link_weight']
        adjacency_matrix[origin, destination] = weight
        adjacency_matrix[destination, origin] = weight  # 如果是无向图，添加对称边

    # 从 geo 中提取指定列
    node = geo[['DISTRICT_ID', 'Area', 'RoadDensity', 'BuildingArea', 
                'CarStation', 'CarPark', 'Subway', 'POI']]

    # 将 DataFrame 转为 NumPy 数组
    node_array = node.values

    return dyna_traffic, time_unix, ext_weather,ind_seq,ind_hor,adjacency_matrix,node_array


def create_sequences(
        dyna_traffic, time_unix, ext_weather, ind_seq, ind_hor,
        sequence_length, forecast_horizon):
    """
    创建用于模型的序列数据。

    参数:
    - dyna_traffic: 交通数据，形状为 [时间步数, 节点数, 特征维度]
    - time_unix: Unix时间戳，形状为 [时间步数, 节点数]
    - ext_weather: 天气数据，形状为 [时间步数, 节点数, 特征维度]
    - ind_seq: 外部指标数据，形状为 [时间步数, 节点数, 特征维度]
    - ind_hor: 未来指标数据，形状为 [时间步数, 节点数, 特征维度]
    - sequence_length: 输入序列的长度
    - forecast_horizon: 预测的时间步数

    返回:
    - dyna_traffic_seq: 交通数据序列，形状为 [样本数, sequence_length, 节点数, 特征维度]
    - time_seq: Unix时间戳序列，形状为 [样本数, sequence_length, 节点数]
    - ext_weather_seq: 天气数据序列，形状为 [样本数, sequence_length, 节点数, 特征维度]
    - ind_seq_seq: 外部指标数据序列，形状为 [样本数, sequence_length, 节点数, 特征维度]
    - ind_hor_seq: 未来指标数据序列，形状为 [样本数, forecast_horizon, 节点数, 特征维度]
    - time_hor: 未来时间戳，形状为 [样本数, forecast_horizon, 节点数]
    - target_seq: 未来交通数据序列，形状为 [样本数, forecast_horizon, 节点数, 特征维度]
    """
    dyna_traffic_seq = []
    time_seq = []
    ext_weather_seq = []
    ind_seq_seq = []
    ind_hor_seq = []
    time_hor = []
    target_seq = []

    # 计算可用的样本数
    total_steps = dyna_traffic.shape[0] - sequence_length - forecast_horizon + 1

    # 遍历所有时间步，创建序列
    for t in range(total_steps):
        # 输入序列：当前时刻 t 到 t + sequence_length
        dyna_traffic_seq.append(dyna_traffic[t:t + sequence_length, :, :])          # [sequence_length, num_nodes, feature_dim]
        time_seq.append(time_unix[t:t + sequence_length, :])                      # [sequence_length, num_nodes]
        ext_weather_seq.append(ext_weather[t:t + sequence_length, :, :])              # [sequence_length, num_nodes, feature_dim]
        ind_seq_seq.append(ind_seq[t:t + sequence_length, :, :])                  # [sequence_length, num_nodes, feature_dim]
        
        # 预测序列：t + sequence_length 到 t + sequence_length + forecast_horizon
        ind_hor_seq.append(ind_hor[t + sequence_length:t + sequence_length + forecast_horizon, :, :])  # [forecast_horizon, num_nodes, feature_dim]
        time_hor.append(time_unix[t + sequence_length:t + sequence_length + forecast_horizon, :])              # [forecast_horizon, num_nodes]
        target_seq.append(dyna_traffic[t + sequence_length:t + sequence_length + forecast_horizon, :, :])          # [forecast_horizon, num_nodes, feature_dim]

    # 将列表转换为 NumPy 数组
    dyna_traffic_seq = np.array(dyna_traffic_seq)    # [num_samples, sequence_length, num_nodes, feature_dim]
    time_seq = np.array(time_seq)                    # [num_samples, sequence_length, num_nodes]
    ext_weather_seq = np.array(ext_weather_seq)      # [num_samples, sequence_length, num_nodes, feature_dim]
    ind_seq_seq = np.array(ind_seq_seq)          # [num_samples, sequence_length, num_nodes, feature_dim]
    ind_hor_seq = np.array(ind_hor_seq)          # [num_samples, forecast_horizon, num_nodes, feature_dim]
    time_hor = np.array(time_hor)                      # [num_samples, forecast_horizon, num_nodes]
    target_seq = np.array(target_seq)                        # [num_samples, forecast_horizon, num_nodes, feature_dim]

    return dyna_traffic_seq, time_seq, ext_weather_seq, ind_seq_seq, ind_hor_seq, time_hor, target_seq


# 自定义 PyTorch Dataset
class TrafficDataset(Dataset):
    def __init__(self, data, stats, time_stats, node, node_stats):
        """
        初始化数据集
        :param data: 数据字典
        :param stats: 包含正则化统计量（均值、标准差、最大值、最小值等）
        :param time_stats: 用于时间编码的时间周期（如24小时和7天）
        :param node: 节点特征数组（形状 [num_nodes, feature_dim]）
        :param node_stats: 节点特征的正则化统计量（均值和标准差）
        """
        self.data = data
        self.stats = stats
        self.time_stats = time_stats
        self.node = node
        self.node_stats = node_stats

    def __len__(self):
        return len(self.data["target_seq"])

    def __getitem__(self, idx):
        # 1. 标准正则化
        dyna_traffic = (self.data["dyna_traffic"][idx] - self.stats["dyna_traffic_mean"]) / self.stats["dyna_traffic_std"]
        ext_weather = (self.data["ext_weather_seq"][idx] - self.stats["ext_weather_mean"]) / self.stats["ext_weather_std"]
        target = (self.data["target_seq"][idx] - self.stats["target_mean"]) / self.stats["target_std"]

        # 2. 时间周期编码
        time_seq = self.data["time_seq"][idx][:, 0]  # [sequence_length]
        time_hor = self.data["time_hor"][idx][:, 0]  # [forecast_horizon]
        time_seq = self.encode_time(time_seq)  # [sequence_length, 4]
        time_hor = self.encode_time(time_hor)  # [forecast_horizon, 4]

        # 扩展到 [sequence_length, num_nodes, 4]
        num_nodes = dyna_traffic.shape[1]
        time_seq = np.expand_dims(time_seq, axis=1).repeat(num_nodes, axis=1)
        time_hor = np.expand_dims(time_hor, axis=1).repeat(num_nodes, axis=1)

        # 最大最小归一化 ind_seq
        ind_seq = (self.data["ind_seq_seq"][idx] - self.stats["ind_seq_min"]) / (self.stats["ind_seq_max"] - self.stats["ind_seq_min"] + 1e-8)
        ind_hor = (self.data["ind_hor_seq"][idx] - self.stats["ind_hor_min"]) / (self.stats["ind_hor_max"] - self.stats["ind_hor_min"] + 1e-8)

        # 4. 节点特征正则化
        node_features = (self.node - self.node_stats["node_mean"]) / self.node_stats["node_std"]

        return {
            "dyna_traffic": torch.tensor(dyna_traffic, dtype=torch.float32),      # [sequence_length, num_nodes, feature_dim]
            "ext_weather": torch.tensor(ext_weather, dtype=torch.float32),        # [sequence_length, num_nodes, feature_dim]
            "target": torch.tensor(target, dtype=torch.float32),                  # [forecast_horizon, num_nodes, feature_dim]
            "time_seq": torch.tensor(time_seq, dtype=torch.float32),              # [sequence_length, num_nodes, 4]
            "time_hor": torch.tensor(time_hor, dtype=torch.float32),              # [forecast_horizon, num_nodes, 4]
            "ind_seq": torch.tensor(ind_seq, dtype=torch.float32),                # [sequence_length, num_nodes, feature_dim]
            "ind_hor": torch.tensor(ind_hor, dtype=torch.float32),                # [forecast_horizon, num_nodes, feature_dim]
            "node": torch.tensor(node_features, dtype=torch.float32),             # [num_nodes, feature_dim]
        }

    def encode_time(self, time_array):
        """
        对时间字段进行周期编码
        :param time_array: 二维时间数组（形状为 [sequence_length or forecast_horizon, num_nodes]）
        :return: 周期编码后的时间特征，形状为 [sequence_length or forecast_horizon, num_nodes, 4]
        """
        # 将 time_array 展平为一维数组
        time_array_flat = time_array.flatten()

        # 使用 's' 作为单位，因为时间戳单位为秒
        unit = 's'

        # 转换为日期时间
        datetime_objs = pd.to_datetime(time_array_flat, unit=unit, errors='coerce')

        # 检查是否存在 NaT 值
        if datetime_objs.isnull().any():
            print("Warning: Some timestamps could not be converted to datetime.")

        # 提取小时和星期几
        hour = datetime_objs.hour
        weekday = datetime_objs.weekday

        # 周期编码
        hour_sin = np.sin(2 * np.pi * hour / self.time_stats["hour_cycle"])
        hour_cos = np.cos(2 * np.pi * hour / self.time_stats["hour_cycle"])
        weekday_sin = np.sin(2 * np.pi * weekday / self.time_stats["weekday_cycle"])
        weekday_cos = np.cos(2 * np.pi * weekday / self.time_stats["weekday_cycle"])

        # 堆叠编码后的特征
        encoded_time_flat = np.stack([hour_sin, hour_cos, weekday_sin, weekday_cos], axis=-1)  # [total_elements, 4]

        # 将编码后的特征重新调整为原始形状
        encoded_time = encoded_time_flat.reshape(time_array.shape + (4,))  # [sequence_length, num_nodes, 4]
        return encoded_time
    

# 数据加载器函数
def create_data_loader(data, stats, time_stats, node, node_stats, batch_size=16, shuffle=True, num_workers=0):
    """
    创建 PyTorch DataLoader。

    参数:
    - data: 数据字典
    - stats: 正则化统计量
    - time_stats: 时间周期信息
    - node: 节点特征数组
    - node_stats: 节点特征的正则化统计量
    - batch_size: 每批数据的大小
    - shuffle: 是否打乱数据
    - num_workers: 数据加载时使用的子进程数

    返回:
    - data_loader: PyTorch DataLoader 对象
    """
    dataset = TrafficDataset(data, stats, time_stats, node, node_stats)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def split_train_test(
    data,             # Data dictionary
    train_val_ratio=0.7   # Training data ratio
):
    num_samples = data["target_seq"].shape[0]
    train_length = int(num_samples * train_val_ratio)

    train_data = {key: value[:train_length] for key, value in data.items()}
    test_data = {key: value[train_length:] for key, value in data.items()}

    return train_data, test_data

def split_train_val_test(
    data,             # Data dictionary
    train_ratio=0.6,  # Training data ratio
    val_ratio=0.1,    # Validation data ratio
    test_ratio=0.3
):
    num_samples = data["target_seq"].shape[0]
    train_length = int(num_samples * train_ratio)
    val_length = int(num_samples * val_ratio)
    test_length = int(num_samples * test_ratio)

    train_data = {key: value[:train_length] for key, value in data.items()}
    val_data = {key: value[train_length:train_length + val_length] for key, value in data.items()}
    test_data = {key: value[train_length + val_length:] for key, value in data.items()}

    return train_data, val_data, test_data




def load(geo,rel,dyna,ext,fut,sequence_length,forecast_horizon,batch_size,num_workers,train_val_ratio, train_ratio, val_ratio):

    dyna_traffic, time_unix, ext_weather,ind_seq,ind_hor,adjacency_matrix,node_array = read_and_process(geo,rel,dyna,ext,fut)

    dyna_traffic_seq, time_seq, ext_weather_seq, ind_seq_seq, ind_hor_seq, time_hor, target_seq = create_sequences(
    dyna_traffic=dyna_traffic,
    time_unix=time_unix,
    ext_weather=ext_weather,
    ind_seq=ind_seq,
    ind_hor=ind_hor,
    sequence_length=sequence_length,
    forecast_horizon=forecast_horizon
    )

    data = {
        "dyna_traffic": dyna_traffic_seq,
        "time_seq": time_seq,
        "ext_weather_seq": ext_weather_seq,
        "ind_seq_seq": ind_seq_seq,
        "ind_hor_seq": ind_hor_seq,
        "time_hor": time_hor,
        "target_seq": target_seq,
    }

    # 计算正则化统计量
    stats = {
        "dyna_traffic_mean": data["dyna_traffic"].mean(axis=0),
        "dyna_traffic_std": data["dyna_traffic"].std(axis=0),
        "ext_weather_mean": data["ext_weather_seq"].mean(axis=0),
        "ext_weather_std": data["ext_weather_seq"].std(axis=0),
        "ind_seq_max": data["ind_seq_seq"].max(axis=0),
        "ind_seq_min": data["ind_seq_seq"].min(axis=0),
        "ind_hor_max": data["ind_hor_seq"].max(axis=0),
        "ind_hor_min": data["ind_hor_seq"].min(axis=0),
        "target_mean": data["target_seq"].mean(axis=0),
        "target_std": data["target_seq"].std(axis=0),
    }

    # 时间周期信息
    time_stats = {
        "hour_cycle": 24,      # 一天24小时
        "weekday_cycle": 7,    # 一周7天
    }

    # 计算节点特征的均值和标准差
    node_stats = {
        "node_mean": node_array.mean(axis=0),  # [feature_dim_node]
        "node_std": node_array.std(axis=0),    # [feature_dim_node]
    }
    train_val_data, test_data = split_train_test(
    data=data,
    train_val_ratio= train_val_ratio
    )

    train_val_loader = create_data_loader(
    data=train_val_data,
    stats=stats,
    time_stats=time_stats,
    node=node_array,
    node_stats=node_stats,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
    )

    
    train_data, val_data, test_data = split_train_val_test(
        data=data,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )


    train_loader = create_data_loader(
        data=train_data,
        stats=stats,
        time_stats=time_stats,
        node=node_array,
        node_stats=node_stats,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = create_data_loader(
        data=val_data,
        stats=stats,
        time_stats=time_stats,
        node=node_array,
        node_stats=node_stats,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = create_data_loader(
        data=test_data,
        stats=stats,
        time_stats=time_stats,
        node=node_array,
        node_stats=node_stats,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_val_loader, train_loader, val_loader, test_loader,adjacency_matrix,stats