import logging
import os
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import shutil
import re
import time
import argparse
import torch

from data import extractData, transformData
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist = pytz.timezone('Asia/Kolkata')
        dt = datetime.fromtimestamp(record.created, tz=ist)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

def logging_setup():
    logging.basicConfig(level=logging.INFO, 
                        filename=os.path.join(os.getcwd(), 'log.log'), 
                        filemode='a', 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging

def get_logger(expt, path, name='log'):
    logger = logging.getLogger(expt)
    handler = logging.FileHandler(os.path.join(path, f'{name}.log'))
    
    # Apply IST formatter
    formatter = ISTFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                              datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def copy_file(src_file, dest_path):
    try:
        if not os.path.isfile(src_file):
            raise FileNotFoundError(f"Source file '{src_file}' not found.")
        
        if os.path.isdir(dest_path):
            dest_file = os.path.join(dest_path, os.path.basename(src_file))
        else:
            dest_file = dest_path
        
        dest_dir = os.path.dirname(dest_file)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        shutil.copy2(src_file, dest_file)
    except Exception as e:
        print(f"An error occurred while copying: {e}")


def plot_metrics(train_value, val_value, save_path, name=''):
    train_epochs, train_values = zip(*train_value)
    if len(val_value) > 0:
        val_epochs, val_values = zip(*val_value)

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_values, label=f'Training {name}', marker='o', linestyle='-')
    if len(val_value) > 0:
        plt.plot(val_epochs, val_values, label=f'Validation {name}', marker='x', linestyle='--')

    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.title(f'Training Vs Validation {name}')
    plt.legend()
    plt.grid(True, axis='y')

    plt.savefig(save_path)
    plt.close()

    return 

def get_max_expt_number(data):
    expt_numbers = []
    for entry in data:
        match = re.search(r'expt-(\d+)', entry)
        if match:
            expt_numbers.append(int(match.group(1)))
    return max(expt_numbers) if expt_numbers else 0

class Watch:
    def __init__(self):
        self.st = time.time()

    def start(self):
        self.st = time.time()
        return 

    def end(self):
        time_consumed = lambda x: round((time.time()-x)/60, 3)
        ts = time_consumed(self.st)
        return f'Done: {ts//60}Hr {ts%60} Mins.'

def prepareParser():
    parser = argparse.ArgumentParser(description='Input configuration')
    parser.add_argument('--config-file', type=str, default='config.yml', help='Configuration file.')
    parser.add_argument('--retrain-model', type=bool, default=False, help='Continue training of the pretrained model.')
    parser.add_argument('--pretrained-model-path', type=str, help='Pretrained saved model path.')

    args = parser.parse_args()
    return args

def testModel(test_dataloader, model, device, criteria, config):
    model.eval()    
    data = next(iter(test_dataloader)).to(device)
    pe = data.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(device)
    out = model(torch.cat([data.x, pe], dim=2), data.edge_index, data.edge_attr).cpu()
    data = data.cpu()   
    print('\nSample test') 
    prob = softmax(out.detach(), dim=0, dtype=torch.float32).numpy().tolist()
    trimmed_prob = [float(f"{num:.5f}") for num in prob]
    print('> model output', trimmed_prob)
    print(f'> predicted class: {out.argmax()}, actual class: {data.y[0]-1}')

    test_loss = 0
    preds, actuals = [], []
    
    ## test model
    for batch in test_dataloader:
        x = batch.x.to(device)
        laplacian_pe = batch.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(device)
        x = torch.cat([x, laplacian_pe], dim=2) # shape: (n_nodes, time, node_features+PE)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        
        out = model(x, edge_index, edge_attr)
        y = batch.y.to(device)[0]-1 
        loss = criteria(out, y)
        test_loss += loss.item()

        preds.append(int(out.argmax()))
        actuals.append(int(batch.y[0])-1)

    return out, test_loss/len(test_dataloader), f1_score(actuals, preds, average='macro')

def load_model(model, optimizer, logger, config):
    checkpoint_path = os.path.join(config['result_dir'], config['training']['pretrained_model_path'])
    assert os.path.exists(checkpoint_path), 'Provided pretrained model does not exists.'

    checkpoint = torch.load(checkpoint_path)
    logger.info('')
    logger.info(f"Loading pretrained model from: {config['training']['pretrained_model_path']}")
    logger.info(f"> Used data info:: names: {checkpoint['data_names']}, range: {checkpoint['case_range']}")
    logger.info(f"> epoch:{checkpoint['epoch']}:: train_loss:{checkpoint['loss']} | train_f1_score: {checkpoint['f1_score']}")

    if config['training']['only_load_attention']:
        model_state = model.state_dict()
        filtered_dict = {}
        for k, v in checkpoint['model'].items():
            if k in model_state and v.shape == model_state[k].shape and ('layers' in k):
                filtered_dict[k] = v
        model_state.update(filtered_dict)
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])

    return model, optimizer

def prepareData(config, logger, rank=None, world_size=None):
    dataset = extractData(data_path=config['dataset']['data_path'],
                          data_names=config['dataset']['data_names'],
                          window_size=config['dataset']['window_size'],
                          n_nodes=config['dataset']['n_nodes'],
                          n_edges=config['dataset']['n_edges'],
                          edge_features=config['dataset']['edge_features'],
                          node_features=config['dataset']['node_features'],
                          case_range=config['dataset']['case_range'], 
                          stride=config['dataset']['stride'],
                          data_portion=config['dataset']['data_portion_from_end'],
                          ignored_fault_locations=config['dataset']['ignored_fault_locations'],
                          task=config['task'],
                          current_as_node_features=config['dataset']['current_as_node_features'])
    
    dataset = transformData(k=config['dataset']['k'], dataset=dataset)
    logger.info(f'Dataset Size: {len(dataset)}')
    train_dataset, test_dataset = train_test_split(dataset, test_size=config['dataset']['test_ratio'], random_state=config['random_seed'])
    
    if config['training']['multi_gpu']:
        # train_dataloader = DataLoader(train_dataset, 
        #                               batch_size=1, 
        #                               num_workers=config['n_workers'], 
        #                               sampler=DistributedSampler(train_dataset, shuffle=True, rank=rank, num_replicas=world_size, drop_last=True))
        # train_dataloader = DataLoader(train_dataset,
        #                         batch_size=1,
        #                         sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True),
        #                         num_workers=config['n_workers'],
        #                         pin_memory=True,
        #                         persistent_workers=True
        #                     )
        data_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=1,
                                    sampler=data_sampler,
                                    num_workers=config['n_workers'],           # ← disable multiprocessing in dataloader
                                    pin_memory=False,        # ← avoid the pin memory crash
                                    drop_last=True
                                )

    else:    
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers'])

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers']) 

    return train_dataloader, test_dataloader  


def get_optimizer(config, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], eps=1e-9)

    if config['training']['l2_regularize']:
        params_decay = [p for n,p in model.named_parameters() if "bias" not in n]
        params_no_decay = [p for n,p in model.named_parameters() if "bias" in n]

        optimizer = torch.optim.AdamW(
            [{"params": params_decay, "weight_decay": config['training']['l2_lambda']},
            {"params": params_no_decay, "weight_decay": 0}],
            lr=config['training']['learning_rate']
        )

    return optimizer