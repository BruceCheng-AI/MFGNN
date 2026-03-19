# 基本库
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy

# PyTorch 相关
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import amp
from torch.optim.lr_scheduler import OneCycleLR
from lion_pytorch import Lion

# 自定义模型和方法
from scripts.FSTGNN import FSTGNN

# 进度条
from tqdm import tqdm

# sklearn 库
from sklearn.metrics import mean_absolute_error, mean_squared_error


def set_random_seed(seed):
    """设置随机种子以保证结果可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sequential_meta_tasks(data_loader, num_tasks=10):
    """
    将数据顺序地分割为 num_tasks 个元任务。
    :param data_loader: DataLoader 对象
    :param num_tasks: 要创建的元任务数量
    :return: 元任务数据集列表，每个元任务是一个样本列表
    """
    # 收集所有样本
    all_samples = []
    for batch in data_loader:
        batch_size = batch["target"].shape[0]
        for i in range(batch_size):
            sample = {key: batch[key][i] for key in batch}
            all_samples.append(sample)
    
    total_samples = len(all_samples)
    task_size = total_samples // num_tasks
    meta_tasks = []
    
    for i in range(num_tasks):
        start_idx = i * task_size
        if i == num_tasks - 1:
            end_idx = total_samples  # 最后一个元任务包含剩余的所有样本
        else:
            end_idx = (i + 1) * task_size
        task_samples = all_samples[start_idx:end_idx]
        meta_tasks.append(task_samples)
    
    return meta_tasks

# 将每个元任务的数据分割为支持集和查询集
def split_task_data(task_data, support_ratio=0.8):
    total_samples = len(task_data)
    split_point = int(total_samples * support_ratio)
    support_data = task_data[:split_point]
    query_data = task_data[split_point:]
    return support_data, query_data

# 自定义的collate_fn
def custom_collate_fn(batch):
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([sample[key] for sample in batch], dim=0)
        else:
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch

# 创建任务加载器
def create_task_loader(task_data, batch_size=16):
    if len(task_data) == 0:
        return None
    class TaskDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    dataset = TaskDataset(task_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return loader

def compute_gradient_cosine_similarity(model):
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.view(-1)
    
    layer_names = list(gradients.keys())
    cosine_similarities = {name: {} for name in layer_names}
    
    for i, name_i in enumerate(layer_names):
        grad_i = gradients[name_i]
        for j, name_j in enumerate(layer_names):
            if i >= j:
                continue  # 避免重复计算
            grad_j = gradients[name_j]
            if grad_i.shape != grad_j.shape:
                continue
            if grad_i.norm() == 0 or grad_j.norm() == 0:
                cos_sim = 0.0
            else:
                cos_sim = torch.nn.functional.cosine_similarity(grad_i, grad_j, dim=0).item()
            cosine_similarities[name_i][name_j] = cos_sim
            cosine_similarities[name_j][name_i] = cos_sim
    return cosine_similarities

def adjust_learning_rates_gcacs(model, optimizer, threshold, scaling_factor,meta_lr):
    cosine_similarities = compute_gradient_cosine_similarity(model)
    
    # 初始化缩放因子
    scaling_factors = {name: 1.0 for name, _ in model.named_parameters()}
    
    for name, similarities in cosine_similarities.items():
        for other_name, sim in similarities.items():
            if sim < threshold:
                scaling_factors[name] *= scaling_factor
    
    # 调整优化器中每个参数组的学习率
    for param_group in optimizer.param_groups:
        name = param_group.get('name', None)
        if name and name in scaling_factors:
            original_lr = meta_lr  # 恢复原始元学习率作为基准
            param_group['lr'] = original_lr * scaling_factors[name]

def meta_learing(train_val_loader, hidden_dim, meta_lr, adapt_steps, epochs, adjacency_matrix, threshold, scaling_factor,device,forecast_horizon):
    # 设置随机种子
    set_random_seed(42)
    # 模型架构参数
    num_heads = 4        # 增加注意力头数以捕获更复杂的关系
    edge_hidden_dim = 32 # 增加边特征维度以更好地建模节点间关系
    num_layers = 2        # 增加层数以提取更深层次的特征
    dropout = 0.1         # 适当增加dropout以防止过拟合

    # 元学习超参数
    max_norm = 1.0        # 适当放宽梯度裁剪以允许更大的参数更新
    task_batch_size = 4   # 增加任务批次大小以提高泛化性

    # 创建训练和测试的元任务数据集
    train_meta_tasks = create_sequential_meta_tasks(train_val_loader, num_tasks=10)

    # 合并两个任务列表：10个原任务和7个星期任务
    all_meta_tasks = train_meta_tasks



    # 创建每个元任务的DataLoader
    task_loaders = {}
    for i, task_data in enumerate(all_meta_tasks):
        task_name = f"task_{i}"
        loader = create_task_loader(task_data)
        if loader is not None:
            task_loaders[task_name] = loader
        else:
            print(f"Meta Task '{task_name}' has no samples and will be skipped.")

    # 分割任务数据为支持集和查询集
    task_support_query = {}
    for i, task_data in enumerate(all_meta_tasks):
        task_name = f"task_{i}"
        support_data, query_data = split_task_data(task_data)
        support_loader = create_task_loader(support_data)
        query_loader = create_task_loader(query_data)
        task_support_query[task_name] = {"support": support_loader, "query": query_loader}

    # 更新任务列表
    tasks = [(v["support"], v["query"]) for v in task_support_query.values()]

    # 获取数据的维度信息
    for batch in train_val_loader:
        sample_time_seq = batch["time_seq"]
        time_dim = sample_time_seq.shape[-1]
        dyna_traffic_seq = batch["dyna_traffic"]
        var_dim_dyna = dyna_traffic_seq.shape[-1]
        ext_weather_seq = batch["ext_weather"]
        var_dim_weather = ext_weather_seq.shape[-1]
        ind_seq_seq = batch["ind_seq"]
        event_dim = ind_seq_seq.shape[-1]
        target_seq = batch["target"]
        target_dim = target_seq.shape[-1]
        node_array = batch["node"]
        node_feature_dim = node_array.shape[-1]
        break  # 只需要获取一次即可

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
    )

    torch.backends.cudnn.enabled = False
    model = model.to(device)
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32).to(device)

    criterion = nn.L1Loss()

    params = list(model.parameters())
    param_groups = []
    for i, param in enumerate(params):
        param_name = f'param_{i}'
        param_groups.append({'params': [param], 'name': param_name})
    meta_optimizer = optim.AdamW(param_groups, lr=meta_lr)

    meta_loss_history = []

    # 创建输出目录
    output_dir = f"../output/Meta-FSTGNN_{hidden_dim}_{forecast_horizon}_{threshold}_{scaling_factor}"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_meta_loss = 0.0
        selected_tasks = random.sample(tasks, min(task_batch_size, len(tasks)))

        for task_idx, (support_loader, query_loader) in enumerate(selected_tasks):
            initial_state = {name: param.clone() for name, param in model.state_dict().items()}
            task_model = FSTGNN(
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
            task_model.load_state_dict(initial_state)

            task_optimizer = optim.AdamW(task_model.parameters(), lr=meta_lr)

            for step in range(adapt_steps):
                combined_loader = itertools.chain(support_loader, query_loader)
                for data_batch in combined_loader:
                    data_batch = {key: value.to(device) for key, value in data_batch.items()}
                    task_optimizer.zero_grad()
                    outputs = task_model(
                        dyna_traffic=data_batch["dyna_traffic"],
                        ext_weather=data_batch["ext_weather"],
                        time_seq=data_batch["time_seq"],
                        ind_seq=data_batch["ind_seq"],
                        ind_hor=data_batch["ind_hor"],
                        adjacency_matrix=adjacency_matrix_tensor,
                        node_array=data_batch["node"],
                        time_hor=data_batch["time_hor"]
                    )
                    loss = criterion(outputs, data_batch["target"])
                    loss.backward()
                    task_optimizer.step()

            for name, param in model.named_parameters():
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.data = param.data - task_model.state_dict()[name]

            total_meta_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        meta_optimizer.step()
        meta_optimizer.zero_grad()

        avg_meta_loss = total_meta_loss / len(selected_tasks)
        meta_loss_history.append(avg_meta_loss)
        print(f"Epoch {epoch}/{epochs}, Meta Loss: {avg_meta_loss:.4f}")

    # 保存训练结果和模型
    df_meta_loss = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Meta_Loss': meta_loss_history,
        'Threshold': threshold,
        'Scaling_Factor': scaling_factor
    })

    csv_filename = os.path.join(output_dir, f'meta_loss_history.csv')
    npy_filename = os.path.join(output_dir, f'meta_loss_history.npy')
    plot_filename = os.path.join(output_dir, f'meta_loss_plot.png')

    df_meta_loss.to_csv(csv_filename, index=False)
    print(f"Meta Loss history saved to '{csv_filename}'.")

    np.save(npy_filename, np.array(meta_loss_history))
    print(f"Meta Loss history saved to '{npy_filename}'.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), meta_loss_history, label='Meta Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Meta Loss')
    plt.title('Meta Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_filename)
    plt.show()
    print(f"Meta Loss plot saved to '{plot_filename}'.")

    # 保存训练好的模型
    model_filename = os.path.join(output_dir, 'trained_model.pth')
    torch.save(model.state_dict(), model_filename)
    print(f"Trained model saved to '{model_filename}'.")

    return model, output_dir


# 创建一个函数来根据场景标记筛选数据索引
def get_indices_by_scenario(indicator_seq):

    indicator_seq = torch.tensor(indicator_seq)
    
    # 计算场景标记
    scenario_mask = (indicator_seq != 0).any(dim=(1, 2))
    indices = torch.nonzero(scenario_mask).squeeze()
    return indices


def make_parameters_leaf(model):
    """确保模型参数是叶子张量并保留梯度"""
    for param in model.parameters():
        param.requires_grad_(True)
        if not param.is_leaf:
            param.retain_grad()

def fine_tune_model(
    model, 
    fine_tune_loader, 
    fast_lr,
    device, 
    adjacency_matrix,
    epochs=1,
    patience=10,
    min_lr=1e-6,
    weight_decay=0.01,
    mixed_precision=False,
):
    # 设置随机种子
    set_random_seed(42)
    learner = deepcopy(model)
    make_parameters_leaf(learner)
    learner.to(device)
    
    criterion = nn.L1Loss()
    
    optimizer = Lion(
        learner.parameters(),
        lr=fast_lr * 0.1,    # Lion通常使用较低学习率
        weight_decay=weight_decay
    )
    
    total_steps = epochs * len(fine_tune_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=fast_lr,
        total_steps=total_steps,
        pct_start=0.2,            # 20%时间用于预热
        div_factor=10.0,          # 初始学习率 = max_lr/10
        final_div_factor=100.0,   # 最终学习率 = max_lr/1000
        anneal_strategy='cos'     # 使用余弦退火策略
    )
    
    scaler = amp.GradScaler() if mixed_precision and device.type == 'cuda' else None
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # 用于记录训练过程
    train_losses = []

    learner.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        # 使用tqdm进度条
        loop = tqdm(fine_tune_loader, desc=f"Epoch {epoch}/{epochs}")
        batch_count = 0
        
        for batch in loop:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            
            with amp.autocast(device_type='cuda', enabled=mixed_precision and device.type == 'cuda'):
                outputs = learner(
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
            
            if mixed_precision and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # 增加梯度裁剪阈值，允许更大的更新
                torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=5.0)
                optimizer.step()
            
            # 在每个batch后更新学习率
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_count += 1
            
            # 更新进度条信息
            loop.set_postfix(loss=batch_loss, lr=current_lr)
        
        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)
        
        # 每个epoch输出综合信息
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_epoch_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping检查
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            best_model_state = learner.state_dict().copy()
            print(f"✓ 保存新的最佳模型 (Loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            print(f"✗ 损失未改善 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                if best_model_state is not None:
                    learner.load_state_dict(best_model_state)
                    print(f"已加载最佳模型 (Loss: {best_loss:.4f})")
                break
    
    # 确保使用最佳模型
    if best_model_state is not None and patience_counter >= patience:
        learner.load_state_dict(best_model_state)
    
    # 打印训练损失趋势
    print(f"训练损失历史: {[round(loss, 4) for loss in train_losses]}")
    
    return learner


def evaluate(learner, test_loader, device, adjacency_matrix,stats):

    learner.eval()

    # 准备目标的均值和标准差用于逆正则化
    target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32).to(device)
    target_std = torch.tensor(stats["target_std"], dtype=torch.float32).to(device)
    
    # 添加 batch 维度，以便在计算中广播
    target_mean = target_mean.unsqueeze(0)  # 形状：[1, forecast_horizon, num_nodes, feature_dim]
    target_std = target_std.unsqueeze(0)

    # 用于存储所有预测值和目标值
    all_outputs = []
    all_targets = []
    
    
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating")
        for batch in loop:
            batch = {key: value.to(device) for key, value in batch.items()}
    
            # 前向传播
            outputs = learner(
                dyna_traffic=batch["dyna_traffic"],
                ext_weather=batch["ext_weather"],
                time_seq=batch["time_seq"],
                ind_seq=batch["ind_seq"],
                ind_hor=batch["ind_hor"],
                adjacency_matrix=adjacency_matrix,
                node_array=batch["node"],
                time_hor=batch["time_hor"]
            )

            targets = batch["target"]
            
            # 逆正则化
            outputs_original = outputs * target_std + target_mean
            target_original = targets * target_std + target_mean
            
            # 将预测值和真实值添加到列表中
            all_outputs.append(outputs_original.cpu())
            all_targets.append(target_original.cpu())  

    # 将所有批次的数据拼接在一起
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 展平数据以计算指标
    outputs_flat = all_outputs.view(-1).numpy()
    targets_flat = all_targets.view(-1).numpy()

    # 计算指标
            
    # 计算指标
    mae = mean_absolute_error(targets_flat, outputs_flat)
    mse = mean_squared_error(targets_flat, outputs_flat)
    rmse = np.sqrt(mse)
    
    # 计算 MAPE (避免除零问题)
    mape = np.mean(np.abs((targets_flat - outputs_flat) / (targets_flat + 0.1))) * 100  # 以百分比表示
    
    # 均保留2位小数
    mae = round(mae, 2)
    rmse = round(rmse, 2)
    mse = round(mse, 2)
    mape = round(mape, 2)

    # 保留两位小数的输出格式
    print(f"Validation Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")

    return mae, mse, rmse, mape,targets_flat,outputs_flat

# 创建一个函数来获取测试数据中某个场景的样本索引
def get_test_indices_by_scenario(indicator_seq):
    indicator_seq = torch.tensor(indicator_seq)
    
    scenario_mask = (indicator_seq != 0).any(dim=(1, 2))
    indices = torch.nonzero(scenario_mask).squeeze()
    return indices

def fine_tune_and_evaluate(model, train_loader, test_loader, scenario_name, fast_lr, device, adjacency_matrix, epochs, stats, output_dir):
    print(f"\nFine-tuning on {scenario_name} Scenario")
    
    # 微调模型
    fine_tuned_learner = fine_tune_model(
        model=model,
        fine_tune_loader=train_loader,
        fast_lr=fast_lr,
        device=device,
        adjacency_matrix=adjacency_matrix,
        epochs=epochs,
        patience=10,
        min_lr=1e-6,
        weight_decay=0.001,
        mixed_precision=False
    )

    # 在测试集上评估
    print(f"\nEvaluating on {scenario_name} Scenario")
    mae, mse, rmse, mape, targets_flat, outputs_flat = evaluate(
        learner=fine_tuned_learner,
        test_loader=test_loader,
        device=device,
        adjacency_matrix=adjacency_matrix,
        stats=stats
    )

    # 创建输出目录
    scenario_output_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    # 保存数据
    targets_flat_filename = os.path.join(scenario_output_dir, 'targets_flat.npy')
    outputs_flat_filename = os.path.join(scenario_output_dir, 'outputs_flat.npy')

    np.save(targets_flat_filename, targets_flat)
    np.save(outputs_flat_filename, outputs_flat)

    print(f"Saved targets_flat to '{targets_flat_filename}'")
    print(f"Saved outputs_flat to '{outputs_flat_filename}'")

    return mae, mse, rmse, mape, targets_flat, outputs_flat

def evaluate2(learner, test_loader, device, adjacency_matrix, stats, return_indices=False):
    learner.eval()

    # 准备目标的均值和标准差用于逆正则化
    target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32).to(device)
    target_std = torch.tensor(stats["target_std"], dtype=torch.float32).to(device)
    
    # 添加 batch 维度，以便在计算中广播
    target_mean = target_mean.unsqueeze(0)  # 形状：[1, forecast_horizon, num_nodes, feature_dim]
    target_std = target_std.unsqueeze(0)

    # 用于存储所有预测值和目标值
    all_outputs = []
    all_targets = []
    all_indices = []  # 用于存储原始数据集中的索引
    
    # 对于 Subset 和 DataLoader，我们可以通过 dataset 的原始索引获取每个样本的原始数据集索引
    # test_loader.dataset 应该是一个 Subset, Subset.dataset 是原始数据集
    # test_loader.dataset.indices 是子集对应原始数据集的索引列表。
    
    subset_indices = test_loader.dataset.indices if hasattr(test_loader.dataset, 'indices') else None

    loop = tqdm(test_loader, desc="Evaluating")
    start_idx = 0
    for batch in loop:
        batch_size = batch["target"].shape[0]
        
        # 获取该batch对应的原始数据集索引（如果是 Subset 则通过 subset_indices[start_idx:start_idx+batch_size]）
        if subset_indices is not None:
            current_indices = subset_indices[start_idx:start_idx+batch_size]
        else:
            # 如果不是subset，则假定是完整数据集，索引即为[start_idx, ...]
            current_indices = list(range(start_idx, start_idx+batch_size))
        
        batch = {key: value.to(device) for key, value in batch.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = learner(
                dyna_traffic=batch["dyna_traffic"],
                ext_weather=batch["ext_weather"],
                time_seq=batch["time_seq"],
                ind_seq=batch["ind_seq"],
                ind_hor=batch["ind_hor"],
                adjacency_matrix=adjacency_matrix,
                node_array=batch["node"],
                time_hor=batch["time_hor"]
            )

        targets = batch["target"]
        
        # 逆正则化
        outputs_original = outputs * target_std + target_mean
        target_original = targets * target_std + target_mean
        
        all_outputs.append(outputs_original.cpu())
        all_targets.append(target_original.cpu())
        all_indices.extend(current_indices)
        
        start_idx += batch_size

    # 将所有批次的数据拼接在一起
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 展平数据以计算整体指标
    outputs_flat = all_outputs.view(-1).numpy()
    targets_flat = all_targets.view(-1).numpy()

    # 计算总体指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    mae = mean_absolute_error(targets_flat, outputs_flat)
    mse = mean_squared_error(targets_flat, outputs_flat)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets_flat - outputs_flat) / (targets_flat + 0.1))) * 100  # 避免除0
    
    # 保留两位小数
    mae = round(mae, 2)
    rmse = round(rmse, 2)
    mse = round(mse, 2)
    mape = round(mape, 2)

    print(f"Validation Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, MAPE: {mape:.2f}%")

    if return_indices:
        return all_outputs, all_targets, all_indices
    else:
        return mae, mse, rmse, mape


def compute_metrics(outputs, targets):
    outputs_flat = outputs.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    
    mae = mean_absolute_error(targets_flat, outputs_flat)
    mse = mean_squared_error(targets_flat, outputs_flat)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets_flat - outputs_flat) / (targets_flat + 0.1))) * 100
    
    mae = round(mae, 2)
    mse = round(mse, 2)
    rmse = round(rmse, 2)
    mape = round(mape, 2)
    
    return mae, mse, rmse, mape

def filter_by_scenario(all_outputs, all_targets, all_indices, scenario_set):
    scenario_mask = [idx in scenario_set for idx in all_indices]
    scenario_mask = np.array(scenario_mask)
    scenario_outputs = all_outputs[scenario_mask]
    scenario_targets = all_targets[scenario_mask]
    return scenario_outputs, scenario_targets



def fine_tune(model,train_val_loader,test_loader,epochs,fast_lr,adjacency_matrix,outputs_dir,stats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32).to(device)
    batch_size = 4

    # 提取测试集中的数据
    test_dataset = test_loader.dataset 

    # 提取预测序列的场景标记 [num_samples, forecast_horizon, num_nodes, 3]
    ind_hor_test = test_dataset.data["ind_hor_seq"]

    # 提取场景标记序列，非0表示异常天气、大型赛事或节假日
    alert_weather_seq_test = ind_hor_test[:, :, :, 0]
    holiday_seq_test = ind_hor_test[:, :, :, 1]
    event_seq_test = ind_hor_test[:, :, :, 2]

    # 提取训练集中的数据
    train_dataset = train_val_loader.dataset 

    # 提取预测序列的场景标记 [num_samples, forecast_horizon, num_nodes, 3]
    ind_hor_train = train_dataset.data["ind_hor_seq"]

    # 提取场景标记序列，非0表示异常天气、大型赛事或节假日
    alert_weather_seq_train = ind_hor_train[:, :, :, 0]
    holiday_seq_train = ind_hor_train[:, :, :, 1]
    event_seq_train = ind_hor_train[:, :, :, 2]

    # 获取异常天气的样本索引
    alert_weather_indices = get_indices_by_scenario(alert_weather_seq_train)

    # 获取节假日的样本索引
    holiday_indices = get_indices_by_scenario(holiday_seq_train)

    # 获取大型赛事的样本索引
    event_indices = get_indices_by_scenario(event_seq_train)

    # 获取异常天气的测试样本索引
    alert_weather_test_indices = get_test_indices_by_scenario(alert_weather_seq_test)

    # 获取节假日的测试样本索引
    holiday_test_indices = get_test_indices_by_scenario(holiday_seq_test)

    # 获取大型赛事的测试样本索引
    event_test_indices = get_test_indices_by_scenario(event_seq_test)

    # 创建测试数据的子集
    alert_weather_test_dataset = Subset(test_dataset, alert_weather_test_indices.tolist())
    holiday_test_dataset = Subset(test_dataset, holiday_test_indices.tolist())
    event_test_dataset = Subset(test_dataset, event_test_indices.tolist())

    # 创建 DataLoader
    alert_weather_test_loader = DataLoader(
        alert_weather_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    holiday_test_loader = DataLoader(
        holiday_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    event_test_loader = DataLoader(
        event_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 获取每个场景的训练样本索引
    alert_weather_indices = get_indices_by_scenario(alert_weather_seq_train)
    holiday_indices = get_indices_by_scenario(holiday_seq_train)
    event_indices = get_indices_by_scenario(event_seq_train)

    # 创建每个场景的微调数据子集
    alert_weather_train_dataset = Subset(train_dataset, alert_weather_indices.tolist())
    holiday_train_dataset = Subset(train_dataset, holiday_indices.tolist())
    event_train_dataset = Subset(train_dataset, event_indices.tolist())

    # 创建每个场景的微调 DataLoader
    alert_weather_train_loader = DataLoader(
        alert_weather_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    holiday_train_loader = DataLoader(
        holiday_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    event_train_loader = DataLoader(
        event_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

  # 在异常天气场景下微调和评估
    fine_tune_and_evaluate(
        model=model,
        train_loader=alert_weather_train_loader,
        test_loader=alert_weather_test_loader,
        scenario_name="Abnormal Weather",
        fast_lr=fast_lr,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        epochs=epochs,
        stats = stats,
        output_dir=outputs_dir
    )


    # 在节假日场景下微调和评估
    fine_tune_and_evaluate(
        model=model,
        train_loader=holiday_train_loader,
        test_loader=holiday_test_loader,
        scenario_name="Holiday",
        fast_lr=fast_lr,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        epochs=epochs,
        stats = stats,
        output_dir=outputs_dir
    )

    # 在大型赛事场景下微调和评估
    fine_tune_and_evaluate(
        model=model,
        train_loader=event_train_loader,
        test_loader=event_test_loader,
        scenario_name="Major Event",

        fast_lr=fast_lr,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        epochs=epochs,
        stats = stats,
        output_dir=outputs_dir
    )

    # 在全量测试集场景下微调和评估（没有筛选的情况）
    full_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    fine_tune_and_evaluate(
        model=model,
        train_loader=train_val_loader,
        test_loader=full_test_loader,
        scenario_name="Full Test Set",
        fast_lr=fast_lr,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        epochs=epochs,
        stats = stats,
        output_dir=outputs_dir
    )

    # 合并场景的训练数据
    all_train_indices = torch.unique(torch.cat([alert_weather_indices, holiday_indices,event_indices]))
    all_train_dataset = Subset(train_dataset, all_train_indices.tolist())

    # 合并场景的测试数据
    all_test_indices = torch.unique(torch.cat([alert_weather_test_indices, holiday_test_indices, event_test_indices]))
    all_test_dataset = Subset(test_dataset, all_test_indices.tolist())

    # 创建合并后的 DataLoader
    all_train_loader = DataLoader(
        all_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    all_test_loader = DataLoader(
        all_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # 微调模型
    fine_tuned_learner = fine_tune_model(
        model=model,
        fine_tune_loader=all_train_loader,
        fast_lr=fast_lr,
        device=device,
        adjacency_matrix=adjacency_matrix_tensor,
        epochs=epochs,
        patience=10,
        min_lr=1e-6,
        weight_decay=0.001,
        mixed_precision=False
    )

    all_outputs, all_targets, all_indices = evaluate2(
        learner=fine_tuned_learner, 
        test_loader=all_test_loader,
        device=device, 
        adjacency_matrix=adjacency_matrix_tensor, 
        stats=stats,
        return_indices=True
    )

    # 展平数据以计算指标
    outputs_flat = all_outputs.view(-1).numpy()
    targets_flat = all_targets.view(-1).numpy()

    # 创建输出目录
    scenario_output_dir = os.path.join(outputs_dir, 'all_events')
    os.makedirs(scenario_output_dir, exist_ok=True)

    # 保存数据
    targets_flat_filename = os.path.join(scenario_output_dir, 'targets_flat.npy')
    outputs_flat_filename = os.path.join(scenario_output_dir, 'outputs_flat.npy')

    np.save(targets_flat_filename, targets_flat)
    np.save(outputs_flat_filename, outputs_flat)  

    # all_indices 是 all_test_dataset 对应于 test_dataset 的原始索引，因为 all_test_dataset 是 test_dataset 的子集
    # 我们已有以下原始场景索引（相对于 test_dataset）：
    # alert_weather_test_indices, holiday_test_indices, event_test_indices

    print('Evaluating on ALL Events Set Scenario')
    alert_set = set(alert_weather_test_indices.tolist())
    holiday_set = set(holiday_test_indices.tolist())
    event_set = set(event_test_indices.tolist())

    # 异常天气场景
    alert_outputs, alert_targets = filter_by_scenario(all_outputs, all_targets, all_indices, alert_set)
    mae, mse, rmse, mape = compute_metrics(alert_outputs, alert_targets)
    print(f"Alert Weather Scenario - MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}%")

    # 节假日场景
    holiday_outputs, holiday_targets = filter_by_scenario(  all_outputs, all_targets, all_indices, holiday_set)
    mae, mse, rmse, mape = compute_metrics(holiday_outputs, holiday_targets)
    print(f"Holiday Scenario - MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}%")

    # 大型赛事场景
    event_outputs, event_targets = filter_by_scenario(all_outputs, all_targets, all_indices, event_set)
    mae, mse, rmse, mape = compute_metrics(event_outputs, event_targets)
    print(f"Event Scenario - MAE: {mae}, RMSE: {rmse}, MSE: {mse}, MAPE: {mape}%")


    # 异常天气场景
    alert_outputs, alert_targets = filter_by_scenario(all_outputs, all_targets, all_indices, alert_set)

    # 分离 traffic_speed 和 TPI
    alert_traffic_outputs = alert_outputs[:, :, :, 0]
    alert_traffic_targets = alert_targets[:, :, :, 0]
    alert_TPI_outputs = alert_outputs[:, :, :, 1]
    alert_TPI_targets = alert_targets[:, :, :, 1]

    # 计算 traffic_speed 的指标
    traffic_mae, traffic_mse, traffic_rmse, traffic_mape = compute_metrics(alert_traffic_outputs, alert_traffic_targets)
    print(f"异常天气场景 - Traffic Speed - MAE: {traffic_mae:.2f}, RMSE: {traffic_rmse:.2f}, MSE: {traffic_mse:.2f}, MAPE: {traffic_mape:.2f}%")

    # 计算 TPI 的指标
    TPI_mae, TPI_mse, TPI_rmse, TPI_mape = compute_metrics(alert_TPI_outputs, alert_TPI_targets)
    print(f"异常天气场景 - TPI - MAE: {TPI_mae:.2f}, RMSE: {TPI_rmse:.2f}, MSE: {TPI_mse:.2f}, MAPE: {TPI_mape:.2f}%")

    # 节假日场景
    holiday_outputs, holiday_targets = filter_by_scenario(all_outputs, all_targets, all_indices, holiday_set)

    # 分离变量
    holiday_traffic_outputs = holiday_outputs[:, :, :, 0]
    holiday_traffic_targets = holiday_targets[:, :, :, 0]
    holiday_TPI_outputs = holiday_outputs[:, :, :, 1]
    holiday_TPI_targets = holiday_targets[:, :, :, 1]

    # 计算 traffic_speed 的指标
    traffic_mae, traffic_mse, traffic_rmse, traffic_mape = compute_metrics(holiday_traffic_outputs, holiday_traffic_targets)
    print(f"节假日场景 - Traffic Speed - MAE: {traffic_mae:.2f}, RMSE: {traffic_rmse:.2f}, MSE: {traffic_mse:.2f}, MAPE: {traffic_mape:.2f}%")

    # 计算 TPI 的指标
    TPI_mae, TPI_mse, TPI_rmse, TPI_mape = compute_metrics(holiday_TPI_outputs, holiday_TPI_targets)
    print(f"节假日场景 - TPI - MAE: {TPI_mae:.2f}, RMSE: {TPI_rmse:.2f}, MSE: {TPI_mse:.2f}, MAPE: {TPI_mape:.2f}%")

    # 大型赛事场景
    event_outputs, event_targets = filter_by_scenario(all_outputs, all_targets, all_indices, event_set)

    # 分离变量
    event_traffic_outputs = event_outputs[:, :, :, 0]
    event_traffic_targets = event_targets[:, :, :, 0]
    event_TPI_outputs = event_outputs[:, :, :, 1]
    event_TPI_targets = event_targets[:, :, :, 1]

    # 计算 traffic_speed 的指标
    traffic_mae, traffic_mse, traffic_rmse, traffic_mape = compute_metrics(event_traffic_outputs, event_traffic_targets)
    print(f"大型赛事场景 - Traffic Speed - MAE: {traffic_mae:.2f}, RMSE: {traffic_rmse:.2f}, MSE: {traffic_mse:.2f}, MAPE: {traffic_mape:.2f}%")

    # 计算 TPI 的指标
    TPI_mae, TPI_mse, TPI_rmse, TPI_mape = compute_metrics(event_TPI_outputs, event_TPI_targets)
    print(f"大型赛事场景 - TPI - MAE: {TPI_mae:.2f}, RMSE: {TPI_rmse:.2f}, MSE: {TPI_mse:.2f}, MAPE: {TPI_mape:.2f}%")

