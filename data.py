import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE, Compose, NormalizeFeatures, ToUndirected

node_features = ['N1_rms[kV]', 'N2_rms[kV]', 'N3_rms[kV]',    # node 1: 0
                'N4_rms[kV]', 'N5_rms[kV]', 'N6_rms[kV]',    # node 2: 1
                'N7_rms[kV]', 'N8_rms[kV]', 'N9_rms[kV]',    # node 3: 2
                'N16_rms[kV]','N17_rms[kV]', 'N18_rms[kV]',  # node 6: 3
                'N19_rms[kV]',  'N20_rms[kV]', 'N21_rms[kV]']# node 7: 4
                
edge_features = ['I121A_rms[kA]', 'I121B_rms[kA]', 'I121C_rms[kA]',  # TL 1-2
                'I232A_rms[kA]', 'I232B_rms[kA]', 'I232C_rms[kA]',  # TL 2-3 
                'I264A_rms[kA]', 'I264B_rms[kA]', 'I264C_rms[kA]',  # TL 2-6
                'I765A_rms[kA]', 'I765B_rms[kA]', 'I765C_rms[kA]',  # TL 7-6
                'S3IA_rms[kA]', 'S3IB_rms[kA]', 'S3IC_rms[kA]']     # TL 3-7

# print('Total node features:', len(node_features))
# print('Total edge features:', len(edge_features))

def extractData(data_path:str, 
                data_names:list, 
                window_size:int, 
                stride:int, 
                n_nodes:int, 
                n_edges:int, 
                n_node_features:int, 
                n_edge_features:int, 
                case_range=None,
                data_portion=0.5):
    
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
                    # [0, 1, 1, 2, 4, 1, 3, 2, 4, 3],
                    # [1, 3, 2, 4, 3, 0, 1, 1, 2, 4]
                    [0, 1, 1, 2, 4],
                    [1, 3, 2, 4, 3]
                ])

    edge_dataset = torch.empty(0, n_edges, window_size, n_edge_features).to(torch.float32)
    node_dataset = torch.empty(0, n_nodes, window_size, n_node_features).to(torch.float32)
    label_dataset = torch.empty(0).to(torch.int8)

    for file in data_files: 
        if current_pattern.fullmatch(file):
            case_number = int(re.findall(f'\d+', file)[-1])

            if (case_range is not None) and (not case_range[0]<=case_number<=case_range[1]):
                continue

            current_df = pd.read_csv(file, index_col=False)
            voltage_df = pd.read_csv(file.replace('ICaseNum', 'VCaseNum'), index_col=False)

            edge_data = torch.tensor(current_df[edge_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())
            edge_data = edge_data[int((1-data_portion)*edge_data.size(0)):]
            edge_data = edge_data.unfold(0, window_size, stride).view(-1, n_edges, n_edge_features, window_size).transpose(-1, -2)#.view(-1, n_edges*window_size, n_edge_features) # (-1, edges, time, edge_features)
            edge_data = edge_data.to(torch.float64)
            # print('Edge data shape:', edge_data.size())

            node_data = torch.tensor(voltage_df[node_features].apply(lambda col: col.fillna(col.mean()), axis=0).to_numpy())
            node_data = node_data[int((1-data_portion)*node_data.size(0)):]

            node_data = node_data\
                .unfold(0, window_size, stride)\
                    .view(-1, n_nodes, n_node_features, window_size)\
                        .transpose(-1, -2) # (-1, nodes, time, node_features)
                            # .contiguous().view(-1, n_nodes*window_size, n_node_features)
            node_data = node_data.to(torch.float64)
            # print('Node data shape:', node_data.size())

            labels = torch.tensor(np.repeat(current_df.FAULT_TYPE.loc[0].astype(int), node_data.size(0), axis=0)) # (-1)
            labels = labels.to(torch.long)
            # print('Label shape:', labels.size())

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

    for i in range(ndata): #, desc='Building Graph Dataset'):
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

