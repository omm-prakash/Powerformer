import os
import torch
import yaml
import pytz
from datetime import datetime
# from tqdm import tqdm

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from torch.nn.functional import softmax
# from torch_geometric.loader import DataLoader

# Distributed training
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.optim.lr_scheduler import StepLR
# # DistributedDataParallel: Use single-machine multi-GPU 
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed import init_process_group, destroy_process_group

from train import trainModel
from train_multi_gpu import init_dpp

from layers import PowerFormer
# from data import extractData, transformData
from utils import *

# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py

def runProcess(config):
    w = Watch()
    w.start()
    ## prepare experiment directory
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    tm = now.strftime('%Y-%m-%d %H:%M')

    if config['task'] == 'detect':
        results = os.path.join(config['result_dir'], 'detect_results')
    elif config['task'] == 'locate':
        results = os.path.join(config['result_dir'], 'locate_results')
    else:
        raise NameError
    
    os.makedirs(results, exist_ok=True)
    entries = os.listdir(results)
    
    expt = "debug" if config['test_mode'] else get_max_expt_number(entries)+1
    result_dir = os.path.join(results, f'expt-{expt}| {tm}') if not config['test_mode'] else os.path.join(config['result_dir'], 'results', 'debug')
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'plots'), exist_ok=True)

    copy_file(os.path.join(os.getcwd(), args.config_file), os.path.join(result_dir, 'config.yml'))
    copy_file(os.path.join(os.getcwd(), 'layers.py'), os.path.join(result_dir, 'layers.py'))
    copy_file(os.path.join(os.getcwd(), 'train.py'), os.path.join(result_dir, 'train(multi-gpu).py'))
    copy_file(os.path.join(os.getcwd(), 'data.py'), os.path.join(result_dir, 'data.py'))

    ## prepare logging setup
    logging = logging_setup()
    logger = get_logger(f'{config['task']}: expt-{expt}', result_dir)

    logger.info('')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(f'++++++++++++++++++++++++++ Experiment: {expt} ++++++++++++++++++++++++++')
    logger.info(f"Description: {config['desc']}")

    logger.info('')
    logger.info('Dataset info.')
    logger.info(f"> Data names: {config['dataset']['data_names']}")
    logger.info(f"> Case range: {config['dataset']['case_range']}")

    logger.info('')
    logger.info('Loading model.')
    ## load model
    model = PowerFormer(d_model=config['model']['d_model'],
                        num_nodes=config['dataset']['n_nodes'],
                        num_edges=config['dataset']['n_edges'],
                        num_heads=config['model']['n_heads'],
                        node_features=config['dataset']['n_node_features']+config['dataset']['k'],
                        edge_features=config['dataset']['n_edge_features'],
                        dropout=config['model']['dropout'],
                        use_bias=config['model']['use_bias'],
                        num_layers=config['model']['n_layers'], 
                        num_fault_types=config['dataset']['num_fault_types'],
                        num_fault_locations=config['dataset']['num_fault_locations'],
                        task = config['task'])

    model = model.to(torch.float64)
    if not config['training']['retrain_model']:
        model.initialize_weights()
        
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"> No. of parameters: {n_params}")
    
    logger.info('')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    if torch.cuda.is_available():
        logger.info(f'> GPU: {torch.cuda.get_device_name(0)}')

    if config['training']['multi_gpu']:
        world_size = torch.cuda.device_count()
        init_dpp(world_size, model, config, logger, result_dir)
    else:
        ## training process 
        train_dataloader, test_dataloader = prepareData(config, logger)
        trainModel(config, train_dataloader, test_dataloader, model, device, logger, result_dir)

    logger.info('-------------------------- Training Complete. --------------------------')
    logger.info(f'{w.end()}')
    logger.info('')

    return

if __name__ == '__main__':
    args = prepareParser()

    with open(args.config_file, 'r') as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    torch.cuda.manual_seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    runProcess(config)
