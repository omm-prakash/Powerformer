import logging
import os
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import shutil
import re
import time
import argparse

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
    val_epochs, val_values = zip(*val_value)

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_values, label=f'Training {name}', marker='o', linestyle='-')
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