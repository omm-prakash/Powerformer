import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, Compose, NormalizeFeatures, ToUndirected

def extractData(data_path:str, 
                data_names:list, 
                window_size:int, 
                stride:int, 
                n_nodes:int, 
                n_edges:int, 
                node_features:list,
                edge_features:list,
                ignored_fault_locations:list,
                task:str,
                current_as_node_features:bool, 
                case_range=None,
                data_portion=0.5):
    assert task in ['detect', 'locate'], f"Invalid task type: {task}"
    csv_file_paths = []
    for data_name in data_names:
        csv_file_paths.append(os.path.join(data_path, data_name))
        print(f'Preparing data from {data_name}')
    
    data_files = [] 
    for csv_file_path in csv_file_paths:
        data_files.extend(list(map(lambda x: os.path.join(csv_file_path, x), os.listdir(csv_file_path))))

    edge_index = torch.tensor([
                                [0, 1, 1, 2, 4],
                                [1, 3, 2, 4, 3]
                            ])

    assert len(node_features)%n_nodes == 0
    assert len(edge_features)%n_edges == 0

    n_edge_features = len(edge_features)//n_edges
    n_node_features = len(node_features)//n_nodes

    edge_dataset = torch.empty(0, n_edges, window_size, n_edge_features).to(torch.float32)
    node_dataset = torch.empty(0, n_nodes, window_size, n_node_features).to(torch.float32)
    label_dataset = torch.empty(0).to(torch.int8)
    pattern = r"case-(\d+)-\(t(\d+)_l(\d+)\)\.csv"

    for file in data_files: 
        # print(file)
        match = re.search(pattern, file)
        if match:
            case_number, fault_type, location = match.groups()            
        else:
            continue  # This avoids crashing
        if int(fault_type) == 1:
            continue

        if int(location) in ignored_fault_locations:
            continue
        try:
            df = pd.read_csv(file, index_col=False)
            df = df.iloc[195:1201]
            
        except:
            continue
        if task == 'detect':
            labels = df["FAULT_TYPE"].to_numpy()
        else:
            labels = df["FAULT_LINE"].to_numpy()
        label_indices = list(range(window_size - 1, len(labels), stride))
        labels = torch.tensor(labels[label_indices], dtype=torch.long)

        edge_data = torch.tensor(df[edge_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())
        edge_data = edge_data.unfold(0, window_size, stride).view(-1, n_edges, n_edge_features, window_size).transpose(-1, -2) # (-1, edges, time, edge_features)
        edge_data = edge_data.to(torch.float64)

        node_data = torch.tensor(df[node_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())
        node_data = node_data\
                            .unfold(0, window_size, stride)\
                                .view(-1, n_nodes, n_node_features, window_size)\
                                    .transpose(-1, -2) # (-1, nodes, time, node_features)
        node_data = node_data.to(torch.float64)

        edge_dataset = torch.concat((edge_dataset, edge_data), dim=0)
        node_dataset = torch.concat((node_dataset, node_data), dim=0)
        label_dataset = torch.concat((label_dataset, labels), dim=0)

    edge_memory = (edge_dataset.element_size() * edge_dataset.nelement()) / (1024 ** 2)
    print(f'\nEdge Dataset: {edge_dataset.size()}, {edge_memory:.2f} MB')
    node_memory = (node_dataset.element_size() * node_dataset.nelement()) / (1024 ** 2)
    print(f'Node Dataset: {node_dataset.size()}, {node_memory:.2f} MB')
    label_memory = (label_dataset.element_size() * label_dataset.nelement()) / (1024 ** 2)
    print(f'Label Dataset: {label_dataset.size()}, {label_memory:.2f} MB')

    ndata = edge_dataset.size(0)
    dataset = []

    for i in range(ndata):
        dataset.append(Data(x=node_dataset[i], edge_index=edge_index, edge_attr=edge_dataset[i], y=label_dataset[i]))

    return dataset


def transformData(k, dataset):

    transforms = Compose([
        ToUndirected(),
        NormalizeFeatures(),
        AddLaplacianEigenvectorPE(k=k, is_undirected=True)
    ])

    dataset = [transforms(data) for data in dataset]
    return dataset

