import torch
import torch.nn as nn
import numpy as np
import os
from copy import deepcopy
from scripts.FSTGNN import FSTGNN
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


def set_random_seed(seed):
    """设置随机种子以保证结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = deepcopy(model.state_dict())
            self.counter = 0

def check_data_for_nan_inf(batch):
    """检查数据中是否有 NaN 或 Inf"""
    for key, value in batch.items():
        if torch.isnan(value).any():
            print(f"NaN detected in {key}")
        if torch.isinf(value).any():
            print(f"Inf detected in {key}")

def validate_model(model, val_loader, criterion, device, adjacency_matrix):
    """验证模型性能"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # 将数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(
                dyna_traffic=batch["dyna_traffic"],
                ext_weather=batch["ext_weather"],
                time_seq=batch["time_seq"],
                ind_seq=batch["ind_seq"],
                ind_hor=batch["ind_hor"],
                adjacency_matrix=adjacency_matrix,
                node_array=batch["node"],
                time_hor=batch["time_hor"]
            )
            
            loss = criterion(outputs, batch["target"])
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, adjacency_matrix, 
                patience=10, min_delta=1e-4):
    """训练模型与自动调参"""
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True)
    
    # 学习率调度器 - ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,  # 每10个epoch降低一次学习率
        gamma=0.7,     # 学习率按这个比例下降
        verbose=True
    )
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            # 获取并移动数据到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(
                dyna_traffic=batch["dyna_traffic"],
                ext_weather=batch["ext_weather"],
                time_seq=batch["time_seq"],
                ind_seq=batch["ind_seq"],
                ind_hor=batch["ind_hor"],
                adjacency_matrix=adjacency_matrix,
                node_array=batch["node"],
                time_hor=batch["time_hor"]
            )
            
            # 检查数值稳定性
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Warning: NaN or Inf detected in outputs")
                continue
                
            # 计算损失
            loss = criterion(outputs, batch["target"])
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            
            # 优化
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        
        # 验证阶段
        val_loss = validate_model(model, val_loader, criterion, device, adjacency_matrix)
        
        # 保存历史记录
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # 加载最佳模型
            model.load_state_dict(early_stopping.best_model)
            break
    
    # 保存训练历史
    return model, training_history



# Function to compute the evaluation metrics
def compute_metrics(outputs, targets):
    mae = mean_absolute_error(targets, outputs)
    mse = mean_squared_error(targets, outputs)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - outputs) / (targets + 0.1))) * 100  # Avoid division by 0
    
    # Round to two decimal places
    mae = round(mae, 2)
    mse = round(mse, 2)
    rmse = round(rmse, 2)
    mape = round(mape, 2)
    
    return mae, mse, rmse, mape

# Define the evaluation function
def evaluate_model_on_scenario(model, test_loader, device, adjacency_matrix, stats, scenario_mask=None):
    """
    Evaluate the model performance on a specific scenario, and compute metrics for traffic speed and TPI.
    If no scenario_mask is provided, it will evaluate the full test set.
    """
    model.eval()
    
    # Prepare target mean and std for inverse normalization
    target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32).to(device)
    target_std = torch.tensor(stats["target_std"], dtype=torch.float32).to(device)
    target_mean = target_mean.unsqueeze(0)  # [1, forecast_horizon, num_nodes, feature_dim]
    target_std = target_std.unsqueeze(0)
    
    all_outputs = []
    all_targets = []
    all_masks = []
    
    current_sample_index = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Scenario"):
            # Get the data and move it to the device
            dyna_traffic = batch["dyna_traffic"].to(device)
            ext_weather = batch["ext_weather"].to(device)
            ind_seq = batch["ind_seq"].to(device)
            ind_hor = batch["ind_hor"].to(device)
            target = batch["target"].to(device)
            time_seq = batch["time_seq"].to(device)
            time_hor = batch["time_hor"].to(device)
            node = batch["node"].to(device)
            
            # Forward pass
            outputs = model(
                dyna_traffic=dyna_traffic,
                ext_weather=ext_weather,
                time_seq=time_seq,
                ind_seq=ind_seq,
                ind_hor=ind_hor,
                adjacency_matrix=adjacency_matrix,
                node_array=node,
                time_hor=time_hor
            )
            
            # Inverse normalization
            outputs_original = outputs * target_std + target_mean
            target_original = target * target_std + target_mean
            batch_size = target.shape[0]
            
            mask_batch = scenario_mask[current_sample_index:current_sample_index + batch_size]
            
            current_sample_index += batch_size
            
            # Append data to the list
            all_outputs.append(outputs_original.cpu().numpy())
            all_targets.append(target_original.cpu().numpy())
            all_masks.append(mask_batch)
    
    # Concatenate all batches of data
    all_outputs = np.concatenate(all_outputs, axis=0)  # [num_samples, forecast_horizon, num_nodes, 2]
    all_targets = np.concatenate(all_targets, axis=0)  # [num_samples, forecast_horizon, num_nodes, 2]
    all_masks = np.concatenate(all_masks, axis=0)      # [num_samples, forecast_horizon, num_nodes]
    
    # 检查 feature_dim
    feature_dim = all_outputs.shape[-1]
    
    if feature_dim == 1:
        all_outputs = np.squeeze(all_outputs, axis=-1)
        all_targets = np.squeeze(all_targets, axis=-1)
    else:
        num_samples, forecast_horizon, num_nodes, feature_dim = all_outputs.shape
        all_outputs = all_outputs.reshape(num_samples, forecast_horizon, num_nodes * feature_dim)
        all_targets = all_targets.reshape(num_samples, forecast_horizon, num_nodes * feature_dim)
        all_masks = np.repeat(all_masks, feature_dim, axis=2)
    
    # 展平数据
    outputs_flat = all_outputs.flatten()
    targets_flat = all_targets.flatten()
    masks_flat = all_masks.flatten()
    
    # 确保维度一致性
    assert outputs_flat.shape[0] == masks_flat.shape[0], "输出和掩码长度不一致"
    
    # 筛选特定场景的样本
    selected_indices = masks_flat == 1
    selected_outputs = outputs_flat[selected_indices]
    selected_targets = targets_flat[selected_indices]
    
    # 计算评估指标
    mae = mean_absolute_error(selected_targets, selected_outputs)
    mse = mean_squared_error(selected_targets, selected_outputs)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((selected_targets - selected_outputs) / (selected_targets + 0.1))) * 100
    
    # 保留两位小数
    mae = round(mae, 2)
    rmse = round(rmse, 2)
    mse = round(mse, 2)
    mape = round(mape, 2)
    
    print(f"Scenario-based Validation Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")
    return mae, rmse, mse, mape

# Define the function to evaluate and save results for all scenarios, including the full test set

def evaluate(model, test_loader, device, adjacency_matrix, stats, alert_weather_mask, holiday_mask, event_mask, full_test_set_mask,all_scenarios_mask, save_folder):
    """
    Evaluate the model performance for all scenarios (including full test set), and save the results to a folder.
    """
    # Evaluate different scenarios
    print("=== Abnormal Weather Scenario ===")
    mae_weather, rmse_weather, mse_weather, mape_weather = evaluate_model_on_scenario(
        model=model,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        stats=stats,
        scenario_mask=alert_weather_mask
    )

    print("\n=== Holiday Scenario ===")
    mae_holiday, rmse_holiday, mse_holiday, mape_holiday = evaluate_model_on_scenario(
        model=model,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        stats=stats,
        scenario_mask=holiday_mask
    )

    print("\n=== Major Event Scenario ===")
    mae_event, rmse_event, mse_event, mape_event = evaluate_model_on_scenario(
        model=model,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        stats=stats,
        scenario_mask=event_mask
    )

    print("\n=== Full Test Set (No Scenario Mask) ===")
    mae_full,rmse_full,mse_full,mape_full = evaluate_model_on_scenario(
        model=model,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        stats=stats,
        scenario_mask=full_test_set_mask  # Pass the full test set mask here
    )

    print("\n=== ALL Events ===")
    mae_all,rmse_all,mse_all,mape_all = evaluate_model_on_scenario(
        model=model,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        stats=stats,
        scenario_mask=all_scenarios_mask  # Pass the full test set mask here
    )



def create_scenario_mask(scenario_sequences):
    """
    创建场景掩码，只要样本在预测时域或节点维度上有任意非零值，就将整个样本标记为该场景
    
    :param scenario_sequences: 场景序列（形状为 [num_samples, forecast_horizon, num_nodes]）
    :return: 场景掩码（形状为 [num_samples, forecast_horizon, num_nodes]）
    """
    # 检查每个样本是否在任何位置包含非零值
    sample_has_scenario = (scenario_sequences != 0).any(axis=(1, 2))  # [num_samples]
    
    # 扩展维度以匹配原始形状
    expanded_mask = sample_has_scenario[:, None, None]  # [num_samples, 1, 1]
    
    # 广播到所有时间步和节点
    full_mask = np.broadcast_to(
        expanded_mask, 
        scenario_sequences.shape
    ).astype(float)
    
    return full_mask

def create_full_test_set_mask(scenario_sequences):
    """
    Create a mask for the full test set where all samples are included (mask is all ones).
    
    :param scenario_sequences: Any scenario sequence (e.g., alert_weather_mask, holiday_mask) 
                               to get the dimensions for the mask.
    :return: Full test set mask (same shape as the input scenario sequence).
    """
    # Create a mask of ones with the same shape as the scenario sequence
    full_mask = np.ones_like(scenario_sequences)
    
    return full_mask



# 示例使用
def train_validate(train_loader, val_loader, test_loader,hidden_dim, num_heads, edge_hidden_dim, num_layers, dropout, adjacency_matrix, num_epochs, random_seed,stats,forecast_horizon):
    # 设置随机种子
    set_random_seed(random_seed)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    for batch in train_loader:
        check_data_for_nan_inf(batch)
        sample_time_seq = batch["time_seq"]  # [batch_size, seq_len, time_dim]
        time_dim = sample_time_seq.shape[-1]
        dyna_traffic_seq = batch["dyna_traffic"]  # [batch_size, seq_len, num_nodes, var_dim_dyna]
        var_dim_dyna = dyna_traffic_seq.shape[-1]
        ext_weather_seq = batch["ext_weather"]  # [batch_size, seq_len, num_nodes, var_dim_weather]
        var_dim_weather = ext_weather_seq.shape[-1]
        ind_seq_seq = batch["ind_seq"]  # [batch_size, seq_len, num_nodes, event_dim]
        event_dim = ind_seq_seq.shape[-1]
        target_seq = batch["target"]  # [batch_size, forecast_horizon, num_nodes, target_dim]
        target_dim = target_seq.shape[-1]
        node_array = batch["node"]  # [batch_size, num_nodes, node_feature_dim]
        node_feature_dim = node_array.shape[-1]
        break  # 只需要获取一次即可

    # 初始化模型
    model = FSTGNN(
        node_feature_dim=node_feature_dim,
        var_dim_dyna=var_dim_dyna,
        var_dim_weather=var_dim_weather,
        event_dim=event_dim,
        time_dim=time_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        target_dim=target_dim,
        edge_hidden_dim=edge_hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 将邻接矩阵移动到设备
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32).to(device)
    
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    
    # 训练模型
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  # 确保已定义验证集加载器
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        patience=5,  # 早停patience
        min_delta=1e-4  # 最小改善阈值
    )
    
    # 创建保存目录
    output_dir = f"../output/FSTGNN_{hidden_dim}_{forecast_horizon}"
    os.makedirs(output_dir, exist_ok=True)

    # Print the output directory path for debugging
    print(f"Output directory created: {output_dir}")

    # 保存模型、历史记录、结果到指定目录
    model_save_path = os.path.join(output_dir, f"model_{hidden_dim}.pth")
    history_save_path = os.path.join(output_dir, f"history_{hidden_dim}.npy")

    
    print(f"Saving model to: {model_save_path}")
    print(f"Saving history to: {history_save_path}")
    
    torch.save(model.state_dict(), model_save_path)
    np.save(history_save_path, history)


    # 提取测试集中的数据
    test_dataset = test_loader.dataset 

    # 提取预测序列的场景标记 [num_samples, forecast_horizon, num_nodes, 3]
    ind_hor_test = test_dataset.data["ind_hor_seq"]

    # 提取场景标记序列，非0表示异常天气、大型赛事或节假日
    alert_weather_seq = ind_hor_test[:, :, :, 0]
    holiday_seq = ind_hor_test[:, :, :, 1]
    event_seq = ind_hor_test[:, :, :, 2]

    # 创建场景掩码
    alert_weather_mask = create_scenario_mask(alert_weather_seq)
    holiday_mask = create_scenario_mask(holiday_seq)
    event_mask = create_scenario_mask(event_seq)
    full_test_set_mask = create_full_test_set_mask(alert_weather_mask)  # Use any mask to get the shape
    # 合并所有场景的掩码
    all_scenarios_mask = np.maximum.reduce([alert_weather_mask, holiday_mask,event_mask])  # 合并场景掩码

    evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        stats=stats,
        alert_weather_mask=alert_weather_mask,
        holiday_mask=holiday_mask,
        event_mask=event_mask,
        full_test_set_mask= full_test_set_mask,
        all_scenarios_mask= all_scenarios_mask,
        save_folder=output_dir
    )


    print(f"Model and training history saved to {output_dir}")
