import os
import torch
# import yaml
# import pytz
# from datetime import datetime
# from tqdm import tqdm

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from torch.nn.functional import softmax
# from torch_geometric.loader import DataLoader

# Distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.optim.lr_scheduler import StepLR
# DistributedDataParallel: Use single-machine multi-GPU 
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed import init_process_group, destroy_process_group


# from layers import GraphTransformer, PowerFormer
# from data import extractData, transformData
from utils import *


def ddp_setup(rank, world_size, logger):
    os.environ["MASTER_ADDR"] = os.getenv("SLURM_NODELIST").split(',')[0]  # or the IP of node 0 if running multi-node
    os.environ["MASTER_PORT"] = "12355"      # choose an open port, same across all processes

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except RuntimeError as e:
        logger.error(f"Failed to initialize process group: {e}")
        raise
    torch.cuda.set_device(rank)

def init_dpp(world_size, model, config, logger, result_dir):
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"

    mp.spawn(ddp_train,
            args=(world_size, model, config, logger, result_dir),
            nprocs=world_size,
            join=True)

def cleanup(logger):
    try:
        dist.barrier()
        dist.destroy_process_group()
    except RuntimeError as e:
        logger.error(f"Failed to destroy process group: {e}")

def ddp_train(rank, world_size, model, config, logger, result_dir):
    
    ddp_setup(rank, world_size, logger)
    dist.barrier()
    train_dataloader, test_dataloader = prepareData(config, logger, rank=rank, world_size=world_size)

    optimizer = get_optimizer(config, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing']).to(rank)

    if config['training']['retrain_model']:
        model, optimizer = load_model(model, optimizer, logger)
    
    logger.info('')
    if config['training']['freeze_attentions']:
        logger.info(f'Freezing attention layers: {config["training"]["freeze_attention_layers"]}')
        model.freeze_attention_layers(config['training']['freeze_attention_layers'])

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_losses, val_losses = [], []
    train_f1_scores, val_f1_scores = [], []
    logger.info('========== Starting model training. ==========')
    for epoch in range(1, config['training']['epochs']+1):
        train_dataloader.sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        epoch_losses = 0
        model.train()        
        preds, actuals, i = [], [], 0
        
        ## train model 
        for batch in train_dataloader:
            x = batch.x.to(rank) # shape: (n_nodes, time, node_features)
            laplacian_pe = batch.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(rank)
            x = torch.cat([x, laplacian_pe], dim=2) # shape: (n_nodes, time, node_features+PE)
            edge_index = batch.edge_index.to(rank) # shape: (2, n_edges)
            edge_attr = batch.edge_attr.to(rank) # shape: (n_edges, time, edge_features)
            y = batch.y.to(rank)[0]-1 

            out = model(x, edge_index, edge_attr) # shape: (num_fault_types,)

            loss = criteria(out, y)
            loss_tensor = torch.tensor(loss.item(), device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_losses += loss_tensor.item() / world_size
            # epoch_losses += loss.item()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if rank==0:
                preds.append(int(out.argmax()))
                actuals.append(int(batch.y[0])-1)
                i += 1
                logger.debug(f'epoch: {epoch+1}-batch: {i+1} :: loss: {loss_tensor.item()}')

        torch.cuda.synchronize()
        if rank==0:
            ## compute metrics
            train_f1_score = f1_score(actuals, preds, average='macro')
            train_loss = epoch_losses/len(train_dataloader)
            logger.info(f'epoch:{epoch}:: train_loss:{train_loss} | train_f1_score: {train_f1_score}')
            train_losses.append((epoch,train_loss))
            train_f1_scores.append((epoch, train_f1_score))

            ## validation step
            if config['training']['validation'] and epoch%config['training']['val_frequency']==0:
                out, test_loss, test_f1_score = testModel(test_dataloader, model, rank, criteria)
                val_losses.append((epoch, test_loss))
                val_f1_scores.append((epoch, test_f1_score))
                logger.info(f'             :: val_loss:{test_loss} | val_f1_score:{test_f1_score}')

            ## save model
            if config['training']['save_model'] and epoch%config['training']['save_frequency']==0 and epoch!=0:
                file = os.path.join(result_dir, 'weights', f'epoch-{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optim': optimizer.state_dict(),
                    'loss': train_loss,
                    'f1_score': train_f1_score, 
                    'expt': result_dir,
                    'data_names': config['dataset']['data_names'],
                    'case_range': config['dataset']['case_range']
                }, f=file)
                logger.info(f'model saved @{file}')

            ## plots metrics
            plot_metrics(train_losses, val_losses, save_path=os.path.join(result_dir, 'plots', 'loss.png'), name='Loss')
            plot_metrics(train_f1_scores, val_f1_scores, save_path=os.path.join(result_dir, 'plots', 'f1_score.png'), name='F1 Score')

    # torch.cuda.empty_cache()
    cleanup(logger)
    return
