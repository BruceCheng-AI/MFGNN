from __future__ import annotations

import torch
import torch.nn as nn


class TimeFeatureEncoder(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.time_fc = nn.Linear(time_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, time_seq: torch.Tensor) -> torch.Tensor:
        return self.activation(self.time_fc(time_seq))


class VariableFeatureEncoder(nn.Module):
    def __init__(self, var_dim: int, hidden_dim: int) -> None:
        super().__init__()
        input_dim = var_dim + hidden_dim
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.activation = nn.ReLU()
        self.residual_fc = nn.Linear(input_dim, hidden_dim)

    def get_sequence(self, var_seq: torch.Tensor, time_encoded: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([var_seq, time_encoded], dim=-1)
        batch_size, seq_len, num_nodes, input_dim = combined.size()
        combined_reshaped = combined.reshape(batch_size * num_nodes, seq_len, input_dim)
        input_fc_output = self.activation(self.input_fc(combined_reshaped))
        gru_output, _ = self.gru(input_fc_output)
        residual = self.residual_fc(combined_reshaped)
        gru_output = gru_output + residual
        return gru_output.reshape(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1, 3)


class EventFeatureEncoder(nn.Module):
    def __init__(self, event_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        input_dim = event_dim + hidden_dim
        self.event_fc = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.activation = nn.ReLU()
        self.residual_fc = nn.Linear(input_dim, hidden_dim)

    def get_sequence(self, event_seq: torch.Tensor, time_encoded: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([event_seq, time_encoded], dim=-1)
        batch_size, seq_len, num_nodes, input_dim = combined.size()
        combined_reshaped = combined.reshape(batch_size * num_nodes, seq_len, input_dim)
        event_encoded = self.activation(self.event_fc(combined_reshaped))
        time_reshaped = time_encoded.reshape(batch_size * num_nodes, seq_len, -1)
        attn_output, _ = self.attention(event_encoded, time_reshaped, time_reshaped)
        residual = self.residual_fc(combined_reshaped)
        attn_output = attn_output + residual
        return attn_output.reshape(batch_size, num_nodes, seq_len, -1).permute(0, 2, 1, 3)


class GraphSAGE(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation if activation is not None else nn.ReLU()
        self.use_identity = in_feats == out_feats
        self.residual_fc = None if self.use_identity else nn.Linear(in_feats, out_feats)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h_agg = torch.bmm(adj, h)
        h_concat = torch.cat([h, h_agg], dim=-1)
        h_out = self.fc(h_concat)
        if self.use_identity:
            h_out = h_out + h
        else:
            h_out = h_out + self.residual_fc(h)
        return self.activation(h_out)


GraphSAGEV = GraphSAGE


class MetaGraphLearner(nn.Module):
    def __init__(self, hidden_dim: int, edge_hidden_dim: int) -> None:
        super().__init__()
        self.node_transform = nn.Linear(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.size()
        transformed = self.node_transform(node_features)
        node_i = transformed.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        node_j = transformed.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_features = torch.cat([node_i, node_j], dim=-1)
        edge_weights = self.edge_mlp(edge_features).squeeze(-1)
        dynamic_adj = torch.sigmoid(edge_weights)
        static_adj = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        return dynamic_adj * static_adj


class GRUFeatureFusion(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, feature_sequences: list[torch.Tensor]) -> torch.Tensor:
        combined = torch.cat(feature_sequences, dim=-1)
        batch_size, seq_len, num_nodes, total_feature_dim = combined.size()
        combined = combined.reshape(batch_size * num_nodes, seq_len, total_feature_dim)
        input_fc_output = self.input_fc(combined)
        gru_output, _ = self.gru(input_fc_output)
        gru_output_last = gru_output[:, -1, :]
        residual = self.residual_fc(combined[:, -1, :])
        fused_feature = self.activation(gru_output_last + residual)
        fused_feature = self.dropout(fused_feature)
        return fused_feature.reshape(batch_size, num_nodes, -1)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.residual_fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, forecast_horizon, num_nodes, input_dim = x.size()
        x = x.transpose(1, 2).contiguous().reshape(batch_size * num_nodes, forecast_horizon, input_dim)
        gru_output, _ = self.gru(x)
        residual = self.residual_fc(x)
        output = self.fc(gru_output + residual)
        return output.reshape(batch_size, num_nodes, forecast_horizon, -1).transpose(1, 2)


class MFGNN(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        var_dim_dyna: int,
        var_dim_weather: int,
        event_dim: int,
        time_dim: int,
        hidden_dim: int,
        num_heads: int,
        target_dim: int,
        edge_hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_encoder = TimeFeatureEncoder(time_dim, hidden_dim)
        self.variable_encoder_dyna = VariableFeatureEncoder(var_dim_dyna, hidden_dim)
        self.variable_encoder_weather = VariableFeatureEncoder(var_dim_weather, hidden_dim)
        self.event_encoder = EventFeatureEncoder(event_dim, hidden_dim, num_heads)
        self.graphsage = GraphSAGE(in_feats=node_feature_dim, out_feats=hidden_dim)
        self.meta_graph_learner = MetaGraphLearner(hidden_dim, edge_hidden_dim)
        self.graphsage_v = GraphSAGEV(in_feats=hidden_dim, out_feats=hidden_dim)
        self.gru_feature_fusion = GRUFeatureFusion(
            input_dim=hidden_dim * 4,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=target_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        dyna_traffic: torch.Tensor,
        ext_weather: torch.Tensor,
        time_seq: torch.Tensor,
        ind_seq: torch.Tensor,
        ind_hor: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        node_array: torch.Tensor,
        time_hor: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _, _ = dyna_traffic.size()

        time_encoded = self.time_encoder(time_seq)
        dyna_encoded_seq = self.variable_encoder_dyna.get_sequence(dyna_traffic, time_encoded)
        ext_weather_encoded_seq = self.variable_encoder_weather.get_sequence(ext_weather, time_encoded)
        ind_seq_encoded_seq = self.event_encoder.get_sequence(ind_seq, time_encoded)

        adj_static = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        node_features_enhanced = self.graphsage(node_array, adj_static)

        dynamic_adj = self.meta_graph_learner(dyna_encoded_seq[:, -1, :, :], adjacency_matrix)
        dyna_encoded_v_seq = []
        for step in range(seq_len):
            dyna_encoded_v = self.graphsage_v(dyna_encoded_seq[:, step, :, :], dynamic_adj)
            dyna_encoded_v_seq.append(dyna_encoded_v.unsqueeze(1))
        dyna_encoded_v_seq = torch.cat(dyna_encoded_v_seq, dim=1)

        node_features_enhanced_seq = node_features_enhanced.unsqueeze(1).repeat(1, seq_len, 1, 1)
        fused_feature = self.gru_feature_fusion(
            [dyna_encoded_v_seq, ext_weather_encoded_seq, ind_seq_encoded_seq, node_features_enhanced_seq]
        )

        time_encoded_hor = self.time_encoder(time_hor)
        ind_hor_encoded_seq = self.event_encoder.get_sequence(ind_hor, time_encoded_hor)
        fused_feature_expanded = fused_feature.unsqueeze(1).expand(-1, time_hor.size(1), -1, -1)
        decoder_input = torch.cat([fused_feature_expanded, ind_hor_encoded_seq], dim=-1)
        return self.decoder(decoder_input)

