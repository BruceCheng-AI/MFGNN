import torch
import torch.nn as nn

# 1. 时间特征编码器
class TimeFeatureEncoder(nn.Module):
    def __init__(self, time_dim, hidden_dim):
        super(TimeFeatureEncoder, self).__init__()
        self.time_fc = nn.Linear(time_dim, hidden_dim)
        self.activation = nn.ReLU()
    
    def forward(self, time_seq, node):
        """
        输入:
            time_seq: [batch_size, seq_len, time_dim]
        输出:
            time_encoded: [batch_size, seq_len, num_nodes, hidden_dim]
        """
        time_encoded = self.activation(self.time_fc(time_seq))  # [batch_size, seq_len, hidden_dim]
        return time_encoded

# 2. 变量特征编码器
class VariableFeatureEncoder(nn.Module):
    def __init__(self, var_dim, hidden_dim):
        super(VariableFeatureEncoder, self).__init__()
        input_dim = var_dim + hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.activation = nn.ReLU()
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
    
    def get_sequence(self, var_seq, time_encoded):
        """
        返回完整的输出序列
        """
        combined = torch.cat([var_seq, time_encoded], dim=-1)  # [batch_size, seq_len, num_nodes, var_dim + hidden_dim]
        batch_size, seq_len, num_nodes, input_dim = combined.size()
        combined_reshaped = combined.view(batch_size * num_nodes, seq_len, input_dim)
        input_fc_output = self.activation(self.input_fc(combined_reshaped))
        gru_output, _ = self.gru(input_fc_output)
        # 残差连接
        residual = self.residual_fc(combined_reshaped)
        gru_output = gru_output + residual
        # 恢复形状
        gru_output = gru_output.view(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1, 3)
        # [batch_size, seq_len, num_nodes, hidden_dim]
        return gru_output

# 3. 事件特征编码器
class EventFeatureEncoder(nn.Module):
    def __init__(self, event_dim, hidden_dim, num_heads):
        super(EventFeatureEncoder, self).__init__()
        input_dim = event_dim + hidden_dim
        self.event_fc = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.activation = nn.ReLU()
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
    
    def get_sequence(self, event_seq, time_encoded):
        combined = torch.cat([event_seq, time_encoded], dim=-1)  # [batch_size, seq_len, num_nodes, event_dim + hidden_dim]
        batch_size, seq_len, num_nodes, input_dim = combined.size()
        combined_reshaped = combined.view(batch_size * num_nodes, seq_len, input_dim)
        event_encoded = self.activation(self.event_fc(combined_reshaped))
        time_encoded_reshaped = time_encoded.reshape(batch_size * num_nodes, seq_len, -1)
        attn_output, _ = self.attention(event_encoded, time_encoded_reshaped, time_encoded_reshaped)
        # 残差连接
        residual = self.residual_fc(combined_reshaped)
        attn_output = attn_output + residual
        # 恢复形状
        attn_output = attn_output.view(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1, 3)
        return attn_output

# 4. 图SAGE
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, out_feats, activation=nn.ReLU()):
        super(GraphSAGE, self).__init__()
        self.fc = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation
        if in_feats == out_feats:
            self.residual = True
        else:
            self.residual = False
            self.residual_fc = nn.Linear(in_feats, out_feats)
    
    def forward(self, h, adj):
        """
        输入:
            h: [batch_size, num_nodes, in_feats]
            adj: [batch_size, num_nodes, num_nodes]
        输出:
            h_out: [batch_size, num_nodes, out_feats]
        """
        h_agg = torch.bmm(adj, h)
        h_concat = torch.cat([h, h_agg], dim=2)
        h_out = self.fc(h_concat)
        if self.residual:
            h_out = h_out + h
        else:
            h_residual = self.residual_fc(h)
            h_out = h_out + h_residual
        h_out = self.activation(h_out)
        return h_out

GraphSAGEV = GraphSAGE

# 5. 元图学习器（Meta Graph Learner）
class MetaGraphLearner(nn.Module):
    def __init__(self, hidden_dim, edge_hidden_dim):
        super(MetaGraphLearner, self).__init__()
        self.node_transform = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1)
        )
    
    def forward(self, node_features, adjacency_matrix):
        # node_features: [batch_size, num_nodes, hidden_dim]
        batch_size, num_nodes, hidden_dim = node_features.size()
        transformed = self.node_transform(node_features)
        node_i = transformed.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = transformed.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_features = torch.cat([node_i, node_j], dim=-1)
        edge_weights = self.edge_mlp(edge_features).squeeze(-1)
        dynamic_adj = torch.sigmoid(edge_weights)
        static_adj = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        dynamic_adj = dynamic_adj * static_adj
        return dynamic_adj

# 6. GRU 特征融合
class GRUFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(GRUFeatureFusion, self).__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, feature_sequences):
        """
        输入:
            feature_sequences: 包含多个形状为 [batch_size, seq_len, num_nodes, hidden_dim] 的张量列表
        输出:
            fused_feature: [batch_size, num_nodes, hidden_dim]
        """
        combined_features = torch.cat(feature_sequences, dim=-1)  # 在特征维拼接
        # [batch_size, seq_len, num_nodes, total_feature_dim]
        batch_size, seq_len, num_nodes, total_feature_dim = combined_features.size()
        combined_features = combined_features.view(batch_size * num_nodes, seq_len, total_feature_dim)
        input_fc_output = self.input_fc(combined_features)
        gru_output, _ = self.gru(input_fc_output)
        gru_output_last = gru_output[:, -1, :]
        residual = self.residual_fc(combined_features[:, -1, :])
        fused_feature = gru_output_last + residual
        fused_feature = self.activation(fused_feature)
        fused_feature = self.dropout(fused_feature)
        fused_feature = fused_feature.view(batch_size, num_nodes, -1)
        return fused_feature

# 7. 解码器
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.residual_fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        """
        输入:
            x: [batch_size, forecast_horizon, num_nodes, input_dim]
        输出:
            output: [batch_size, forecast_horizon, num_nodes, output_dim]
        """
        batch_size, forecast_horizon, num_nodes, input_dim = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size * num_nodes, forecast_horizon, input_dim)
        gru_output, _ = self.gru(x)
        residual = self.residual_fc(x)
        gru_output = gru_output + residual
        output = self.fc(gru_output)
        output = output.view(batch_size, num_nodes, forecast_horizon, -1).transpose(1, 2)
        return output

# 8. 主模型
class FSTGNN_noF(nn.Module):
    def __init__(self, node_feature_dim, var_dim_dyna, var_dim_weather, event_dim, time_dim, hidden_dim,
                 num_heads, target_dim, edge_hidden_dim, num_layers=1, dropout=0.1):
        super(FSTGNN_noF, self).__init__()
        self.time_encoder = TimeFeatureEncoder(time_dim, hidden_dim)
        self.variable_encoder_dyna = VariableFeatureEncoder(var_dim_dyna, hidden_dim)
        self.variable_encoder_weather = VariableFeatureEncoder(var_dim_weather, hidden_dim)
        self.event_encoder = EventFeatureEncoder(event_dim, hidden_dim, num_heads)
        self.graphsage = GraphSAGE(in_feats=node_feature_dim, out_feats=hidden_dim)
        self.meta_graph_learner = MetaGraphLearner(hidden_dim, edge_hidden_dim)
        self.graphsage_v = GraphSAGEV(in_feats=hidden_dim, out_feats=hidden_dim)
        # 添加节点增强特征作为额外序列，故融合input_dim=hidden_dim*4
        self.gru_feature_fusion = GRUFeatureFusion(input_dim=hidden_dim * 4, hidden_dim=hidden_dim,
                                                   num_layers=num_layers, dropout=dropout)
        self.decoder = Decoder(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=target_dim,
                               num_layers=num_layers, dropout=dropout)
    
    def forward(self, dyna_traffic, ext_weather, time_seq, ind_seq, ind_hor, adjacency_matrix, node_array, time_hor):
        """
        输入:
            dyna_traffic: [batch_size, seq_len, num_nodes, var_dim]
            ext_weather: [batch_size, seq_len, num_nodes, ext_weather_dim]
            time_seq: [batch_size, seq_len, time_dim]
            ind_seq: [batch_size, seq_len, num_nodes, event_dim]
            ind_hor: [batch_size, forecast_horizon, num_nodes, event_dim]
            adjacency_matrix: [num_nodes, num_nodes]
            node_array: [batch_size, num_nodes, node_feature_dim]
            time_hor: [batch_size, forecast_horizon, time_dim]
        输出:
            output: [batch_size, forecast_horizon, num_nodes, target_dim]
        """
        batch_size, seq_len, num_nodes, _ = dyna_traffic.size()
        # 编码时间序列
        time_encoded = self.time_encoder(time_seq, num_nodes)
        # 编码各类序列特征
        dyna_encoded_seq = self.variable_encoder_dyna.get_sequence(dyna_traffic, time_encoded)
        ext_weather_encoded_seq = self.variable_encoder_weather.get_sequence(ext_weather, time_encoded)
        ind_seq_encoded_seq = self.event_encoder.get_sequence(ind_seq, time_encoded)
        
        # 节点特征与初始图增强
        node_features = node_array  # [batch_size, num_nodes, node_feature_dim]
        adj_static = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        node_features_enhanced = self.graphsage(node_features, adj_static)  # [batch_size, num_nodes, hidden_dim]

        # 使用 dyna_encoded_seq 的最后一帧特征学习动态邻接矩阵
        dynamic_adj = self.meta_graph_learner(dyna_encoded_seq[:, -1, :, :], adjacency_matrix)
        
        # 使用动态邻接矩阵对 dyna_encoded_seq 做图聚合
        dyna_encoded_v_seq = []
        for t in range(seq_len):
            dyna_encoded_v = self.graphsage_v(dyna_encoded_seq[:, t, :, :], dynamic_adj)
            dyna_encoded_v_seq.append(dyna_encoded_v.unsqueeze(1))
        dyna_encoded_v_seq = torch.cat(dyna_encoded_v_seq, dim=1)

        # 将 node_features_enhanced 扩展至序列长度，以匹配其他序列
        # 此处重复为 seq_len，与 dyna_encoded_v_seq 等保持一致
        node_features_enhanced_seq = node_features_enhanced.unsqueeze(1).repeat(1, seq_len, 1, 1)

        # 特征融合
        feature_sequences = [dyna_encoded_v_seq, ext_weather_encoded_seq, ind_seq_encoded_seq, node_features_enhanced_seq]
        fused_feature = self.gru_feature_fusion(feature_sequences)  # [batch_size, num_nodes, hidden_dim]
        
        # 编码预测时间步的时间特征与事件特征
        time_encoded_hor = self.time_encoder(time_hor, num_nodes)
        ind_hor_encoded_seq = torch.zeros_like(time_encoded_hor) 

        # 解码输入
        fused_feature_expanded = fused_feature.unsqueeze(1).expand(-1, time_hor.size(1), -1, -1)

        # 保持 ind_hor_encoded_seq，拼接它与 fused_feature_expanded
        decoder_input = torch.cat([fused_feature_expanded, ind_hor_encoded_seq], dim=-1)

        # 解码预测
        output = self.decoder(decoder_input)

        return output
