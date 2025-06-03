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
import numpy as np
from data import extractData, transformData
from sklearn.model_selection import train_test_split
from torch.nn.functional import softmax
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix

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
                          stride=config['dataset']['stride'],
                          n_nodes=config['dataset']['n_nodes'],
                          n_edges=config['dataset']['n_edges'],
                          node_features=config['dataset']['node_features'],
                          edge_features=config['dataset']['edge_features'],
                          ignored_fault_locations=config['dataset']['ignored_fault_locations'],
                          task=config['task'],
                          edge_index=config['dataset']['edge_index'],
                          ignored_fault_types=config['dataset']['ignored_fault_types'],
                          case_range=config['dataset']['case_range'])
    
    dataset = transformData(k=config['dataset']['k'], dataset=dataset)
    logger.info(f'Dataset Size: {len(dataset)}')
    train_dataset, test_dataset = train_test_split(dataset, test_size=config['dataset']['test_ratio'], random_state=config['random_seed'])
    
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

def plot_confusion_matrix(preds, actuals, plot_path):

    # Convert to numpy if torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(actuals, preds)
    num_classes = cm.shape[0]

    # Create the heatmap
    fig, ax = plt.subplots()
    heatmap = ax.pcolormesh(cm, cmap='Blues', edgecolors='k', linewidth=1)

    # Add colorbar
    plt.colorbar(heatmap, ax=ax)

    # Set ticks and labels
    tick_labels = [str(i+1) for i in range(num_classes)]
    ax.set_xticks(np.arange(num_classes) + 0.5)
    ax.set_yticks(np.arange(num_classes) + 0.5)
    ax.set_xticklabels(tick_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(tick_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')

    # Annotate each cell with the numeric value (bold and large)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')
    # Adjust layout and save
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(plot_path,"confusion_matrix.png"))
    plt.close()