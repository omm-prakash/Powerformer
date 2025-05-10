import torch
import torch.nn as nn
from torch_geometric.utils import softmax
import torch.nn.init as init

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_dim, in_dim*2)
        self.linear2 = nn.Linear(in_dim*2, in_dim)

    def forward(self, x):
        # expected shape:: x: (*, dim) 
        x = self.linear1(x) # x: (*, 2*dim) 
        x = self.relu(x) # x: (*, 2*dim)
        x = self.drop(x) # x: (*, 2*dim)
        x = self.linear2(x) # x: (*, dim)

        return x # x: (*, dim)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ffn = FeedForwardNN(in_dim, dropout)
        self.norm = nn.LayerNorm(normalized_shape=in_dim)

    def forward(self, x):
        # expected shape:: x: (*, dim) 
        x_in = x 
        x = self.ffn(x) # x: (*, dim)
        x = x + x_in # x: (*, dim)
        x = self.norm(x) # x: (*, dim)

        return x # x: (*, dim)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, node_features, edge_features, use_bias, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert d_model%num_heads==0, 'd_model must be divisible by #heads'

        self.d_model = d_model
        self.heads = num_heads 
        self.k_d = d_model//num_heads
        self.Wq = nn.Linear(node_features, d_model, bias=use_bias)
        self.Wk = nn.Linear(node_features, d_model, bias=use_bias)
        self.Wv = nn.Linear(node_features, d_model, bias=use_bias)
        self.We = nn.Linear(edge_features, d_model, bias=use_bias)

    def forward(self, x, edge_index, edge_attr):
        """
        expected shape
        x: (n_nodes, time, node_features)
        edge_index: (2, n_edges)
        edge_attr: (n_edges, time, edge_features)
        """
        interval = x.size(1)

        Qh = self.Wq(x) # (n_nodes, time, d_model)
        Kh = self.Wk(x) # (n_nodes, time, d_model)
        Vh = self.Wv(x) # (n_nodes, time, d_model)
        Ee = self.We(edge_attr) # (n_edges, time, d_model)

        Qh = torch.transpose(Qh.view(-1, interval, self.heads, self.k_d), 1, 2) # (n_nodes, heads, time, k_d)
        Kh = torch.transpose(Kh.view(-1, interval, self.heads, self.k_d), 1, 2) # (n_nodes, heads, time, k_d)
        Vh = torch.transpose(Vh.view(-1, interval, self.heads, self.k_d), 1, 2) # (n_nodes, heads, time, k_d)
        Ee = torch.transpose(Ee.view(-1, interval, self.heads, self.k_d), 1, 2) # (n_edges, heads, time, k_d)

        Ee = Ee @ torch.transpose(Ee, -1, -2) # (n_edges, heads, time, time)

        score = (Qh[edge_index[1]] @ torch.transpose(Kh[edge_index[0]],-1,-2)) # (n_edges, heads, time, time)
        score = score / torch.sqrt(torch.tensor(self.k_d, dtype=torch.float32, requires_grad=False)) # (n_edges, heads, time, time)
        score = score * Ee # (n_edges, heads, time, time)
        score = score.view(score.size(0), self.heads, -1) # (n_edges, heads, time*time)
        score = softmax(score, edge_index[1], dim=0) # (n_edges, heads, time*time)
        score = score.view(-1, self.heads, interval, interval) # (n_edges, heads, time, time)

        x = torch.zeros_like(Vh) # (n_nodes, heads, time, k_d)
        score = score @ Vh[edge_index[1]] # (n_edges, heads, time, k_d)
        x.index_add_(dim=0, index=edge_index[1], source=score, alpha=1) # (n_nodes, heads, time, k_d)
        x = x.transpose(1,2).contiguous().view(x.size(0),-1,self.d_model) # (n_nodes, time, d_model)

        e = score.transpose(1,2).contiguous().view(score.size(0), -1, self.d_model) # (n_edges, time, d_model)

        return x,e

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, node_features, edge_features, dropout, use_bias, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.Oh = nn.Linear(d_model, node_features) 
        self.norm = nn.LayerNorm(normalized_shape=node_features)
        self.residual = ResidualBlock(node_features, dropout)        

        self.Oh_e = nn.Linear(d_model, edge_features) 
        self.norm_e = nn.LayerNorm(normalized_shape=edge_features)
        self.residual_e = ResidualBlock(edge_features, dropout)        

        self.attention = MultiHeadAttentionLayer(d_model, num_heads, node_features, edge_features, use_bias)

    def forward(self, x, edge_index, edge_attr):
        """
        expected shape
        x: (n_nodes, time, node_features)
        edge_index: (2, n_edges)
        edge_attr: (n_edges, time, edge_features)
        """
        
        x_in = x # (n_nodes, time, node_features)
        e_in = edge_attr # (n_edges, time, edge_features)
        x,e = self.attention(x, edge_index, edge_attr) # (n_nodes, time, d_model), (n_edges, time, d_model)

        x = self.Oh(x) # (n_nodes, time, node_features)
        x = x + x_in # (n_nodes, time, node_features)
        x = self.norm(x) # (n_nodes, time, node_features)
        x = self.residual(x) # (n_nodes, time, node_features)

        e = self.Oh_e(e) # (n_edges, time, edge_features)
        e = e + e_in # (n_edges, time, edge_features)
        e = self.norm_e(e) # (n_edges, time, edge_features)
        e = self.residual_e(e) # (n_edges, time, edge_features)

        return x,e 
    
class GraphTransformer(nn.Module):
    def __init__(self, d_model, num_nodes, num_heads, node_features, edge_features, dropout, use_bias, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList([
            GraphTransformerLayer( d_model, num_heads, node_features, edge_features, dropout, use_bias)
                for _ in range(num_layers)
        ])

        self.linear = nn.Linear(node_features, d_model)
        self.norm = nn.LayerNorm(num_nodes)
    
    def forward(self, x, edge_index, edge_attr):

        for layer in self.layers:
            x,e = layer(x, edge_index, edge_attr) # (n_nodes, time, node_features), (n_edges, time, edge_features)

        x = x.mean(dim=1) # (n_nodes, node_features)
        x = self.linear(x) # (n_nodes, d_model)
        # x = torch.sum(x, dim=-1) # (n_nodes,)
        x = x.mean(dim=-1) # (n_nodes,)
        x = self.norm(x) # (n_nodes,)

        e = e.mean(dim=1) # (n_edges, node_features)
        e = self.linear(e) # (n_edges, d_model)
        e = e.mean(dim=-1) # (n_edges,)
        e = self.norm(e) # (n_edges,)

        return x,e
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming Initialization for ReLU and SiLU activations
                if isinstance(m, (nn.ReLU, nn.SiLU)):
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    init.xavier_normal_(m.weight)
                
                if m.bias is not None:
                    init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        return

    def freeze_attention_layers(self, layers):
        for name, param in self.named_parameters():
            for l in layers:
                if f'layers.{l}' in name:
                    param.requires_grad = False

        return

class PowerFormer(GraphTransformer):
    def __init__(self, d_model, num_nodes, num_edges, num_heads, node_features, edge_features, dropout, use_bias, num_layers, num_fault_types, num_fault_locations, task, *args, **kwargs):
        super().__init__(d_model, num_nodes, num_heads, node_features, edge_features, dropout, use_bias, num_layers, *args, **kwargs)

        # assert task in ['detect', 'locate'], f"Invalid task type: {task}"
        
        # if task == 'detect':
        #     self.num_classes = num_fault_types
        # else:
        #     self.num_classes = num_fault_locations

        self.out_layer = nn.Sequential(
            nn.Linear(num_nodes, 2*num_nodes),
            nn.SiLU(),
            nn.Linear(2*num_nodes, num_nodes),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(num_nodes, num_fault_types)
        )

        self.out_layer_e = nn.Sequential(
            nn.Linear(num_edges, 2*num_edges),
            nn.SiLU(),
            nn.Linear(2*num_edges, num_edges),
            nn.Dropout(dropout),
            nn.SiLU(),
            nn.Linear(num_edges, num_fault_locations)
        )

    def forward(self, x, edge_index, edge_attr):
        x,e = super().forward(x, edge_index, edge_attr) # (n_nodes,), (n_edges,)
        x = self.out_layer(x) # (num_fault_types,)
        e = self.out_layer_e(e) # (num_fault_locations,)

        return x,e
    
    def initialize_weights(self):
        return super().initialize_weights()
    
    def freeze_attention_layers(self, layers):
        return super().freeze_attention_layers(layers)


# class PowerFormerLocate(GraphTransformer):
#     def __init__(self, d_model, num_nodes, num_heads, node_features, edge_features, dropout, use_bias, num_layers, num_fault_types, *args, **kwargs):
#         super().__init__(d_model, num_nodes, num_heads, node_features, edge_features, dropout, use_bias, num_layers, *args, **kwargs)

#         self.out_layer = nn.Sequential(
#             nn.Linear(num_nodes, 2*num_nodes),
#             nn.SiLU(),
#             nn.Linear(2*num_nodes, num_nodes),
#             nn.Dropout(dropout),
#             nn.SiLU(),
#             nn.Linear(num_nodes, num_fault_types)
#         )

#     def forward(self, x, edge_index, edge_attr):
#         x = super().forward(x, edge_index, edge_attr) # (n_nodes,)
#         x = self.out_layer(x) # (num_fault_types,)

#         return x
    
#     def initialize_weights(self):
#         return super().initialize_weights()
    
#     def freeze_attention_layers(self, layers):
#         return super().freeze_attention_layers(layers)

            

        






        





