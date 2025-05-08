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

    current_pattern = re.compile(r'.*/ICaseNum_rms_\d+Results\.csv')
    # voltage_pattern = re.compile(r'VCaseNum_rms_\d+Results.csv')
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

    for file in data_files: 
        if current_pattern.fullmatch(file):
            case_number = int(re.findall(f'\d+', file)[-1])
            if (case_range is not None) and (not case_range[0]<=case_number<=case_range[1]):
                continue

            current_df = pd.read_csv(file, index_col=False)
            fault_location = current_df.FAULT_LINE.loc[0].astype(int)   
            fault_type = current_df.FAULT_TYPE.loc[0].astype(int)

            if fault_location in ignored_fault_locations:
                continue

            complete_edge_data = torch.tensor(current_df[edge_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())
            steady_state_edge_data = complete_edge_data[:100]
            steady_state_edge_data = steady_state_edge_data.unfold(0, window_size, stride).view(-1, n_edges, n_edge_features, window_size).transpose(-1, -2)
            steady_state_edge_data = steady_state_edge_data.to(torch.float64)

            edge_data = complete_edge_data[int((1-data_portion)*complete_edge_data.size(0)):]
            edge_data = edge_data.unfold(0, window_size, stride).view(-1, n_edges, n_edge_features, window_size).transpose(-1, -2) # (-1, edges, time, edge_features)
            edge_data = edge_data.to(torch.float64)
            # print('Edge data shape:', edge_data.size())

            if current_as_node_features:
                missing_columns = [col for col in node_features if col not in current_df.columns]
                if missing_columns:
                    raise ValueError(f"The following node_features columns are missing in current_df: {missing_columns}")
                complete_node_data = torch.tensor(current_df[node_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())

            else:
                voltage_df = pd.read_csv(file.replace('ICaseNum', 'VCaseNum'), index_col=False)
                missing_columns = [col for col in node_features if col not in voltage_df.columns]
                if missing_columns:
                    raise ValueError(f"The following node_features columns are missing in voltage_df: {missing_columns}")
                complete_node_data = torch.tensor(voltage_df[node_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())

            steady_state_node_data = complete_node_data[:100]
            steady_state_node_data = steady_state_node_data.unfold(0, window_size, stride).view(-1, n_nodes, n_node_features, window_size).transpose(-1, -2)
            steady_state_node_data = steady_state_node_data.to(torch.float64) 

            node_data = complete_node_data[int((1-data_portion)*complete_node_data.size(0)):]

            node_data = node_data\
                .unfold(0, window_size, stride)\
                    .view(-1, n_nodes, n_node_features, window_size)\
                        .transpose(-1, -2) # (-1, nodes, time, node_features)
            node_data = node_data.to(torch.float64)
            # print('Node data shape:', node_data.size())

            if task == 'detect':
                steady_state_labels = torch.tensor(np.repeat(1, steady_state_node_data.size(0), axis=0)).to(torch.long)
                labels = torch.tensor(np.repeat(fault_type, node_data.size(0), axis=0))
            else:
                steady_state_labels = torch.tensor(np.repeat(8, steady_state_node_data.size(0), axis=0)).to(torch.long)
                if fault_type == 1:
                    fault_location = 8
                labels = torch.tensor(np.repeat(fault_location, node_data.size(0), axis=0))

            # labels = torch.tensor(np.repeat(fault_type, node_data.size(0), axis=0)) # (-1)
            # labels = labels.to(torch.long)
            # print('Label shape:', labels.size())

            edge_dataset = torch.concat((edge_dataset, edge_data), dim=0)
            node_dataset = torch.concat((node_dataset, node_data), dim=0)
            label_dataset = torch.concat((label_dataset, labels), dim=0)

            edge_dataset = torch.concat((edge_dataset, steady_state_edge_data), dim=0)
            node_dataset = torch.concat((node_dataset, steady_state_node_data), dim=0)
            label_dataset = torch.concat((label_dataset, steady_state_labels), dim=0)

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

