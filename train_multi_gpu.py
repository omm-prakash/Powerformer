import os
import torch

# Distributed training
import torch.distributed as dist
import torch.multiprocessing as mp

# DistributedDataParallel: Use single-machine multi-GPU 
from torch.nn.parallel import DistributedDataParallel as DDP

# from layers import GraphTransformer, PowerFormer
# from data import extractData, transformData
from utils import *


# def ddp_setup(rank, world_size, logger):
#     os.environ["MASTER_ADDR"] = os.getenv("SLURM_NODELIST").split(',')[0]  # or the IP of node 0 if running multi-node
#     os.environ["MASTER_PORT"] = "12355"      # choose an open port, same across all processes
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#     try:
#         torch.cuda.set_device(rank)
#         dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     except RuntimeError as e:
#         logger.error(f"Failed to initialize process group: {e}")
#         raise
#     torch.cuda.set_device(rank)

# def init_dpp(world_size, model, config, logger, result_dir):
#     assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
#     mp.set_start_method("fork", force=True)
#     mp.spawn(ddp_train,
#             args=(world_size, model, config, logger, result_dir),
#             nprocs=world_size,
#             join=True)

# def cleanup(logger, rank):
#     try:
#         # dist.barrier()
#         dist.barrier(device_ids=[rank])
#         dist.destroy_process_group()
#     except RuntimeError as e:
#         logger.error(f"Failed to destroy process group: {e}")

# def ddp_train(rank, world_size, model, config, logger, result_dir):
    
#     ddp_setup(rank, world_size, logger)
#     dist.barrier()
#     train_dataloader, test_dataloader = prepareData(config, logger, rank=rank, world_size=world_size)
#     logger.info(f"[Rank {rank}] Using device: {torch.cuda.current_device()}")

#     optimizer = get_optimizer(config, model)
#     criteria = torch.nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing']).to(rank)

#     if config['training']['retrain_model']:
#         model, optimizer = load_model(model, optimizer, logger)
    
#     logger.info('')
#     if config['training']['freeze_attentions']:
#         logger.info(f'Freezing attention layers: {config["training"]["freeze_attention_layers"]}')
#         model.freeze_attention_layers(config['training']['freeze_attention_layers'])

#     model.to(rank)
#     # model = DDP(model, device_ids=[rank])
#     model = DDP(model, device_ids=[rank], find_unused_parameters=False)


#     train_losses, val_losses = [], []
#     train_f1_scores, val_f1_scores = [], []
#     logger.info('========== Starting model training. ==========')
#     for epoch in range(1, config['training']['epochs']+1):
#         train_dataloader.sampler.set_epoch(epoch)
#         torch.cuda.empty_cache()
#         epoch_losses = 0
#         model.train()        
#         preds, actuals, i = [], [], 0
        
#         ## train model 
#         for batch in train_dataloader:
#             x = batch.x.to(rank) # shape: (n_nodes, time, node_features)
#             laplacian_pe = batch.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(rank)
#             x = torch.cat([x, laplacian_pe], dim=2) # shape: (n_nodes, time, node_features+PE)
#             edge_index = batch.edge_index.to(rank) # shape: (2, n_edges)
#             edge_attr = batch.edge_attr.to(rank) # shape: (n_edges, time, edge_features)
#             y = batch.y.to(rank)[0]-1 

#             out = model(x, edge_index, edge_attr) # shape: (num_fault_types,)

#             loss = criteria(out, y)
#             loss_tensor = torch.tensor(loss.item(), device=rank)
#             dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
#             epoch_losses += loss_tensor.item() / world_size
#             # epoch_losses += loss.item()
            
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()

#             if rank==0:
#                 preds.append(int(out.argmax()))
#                 actuals.append(int(batch.y[0])-1)
#                 i += 1
#                 logger.debug(f'epoch: {epoch+1}-batch: {i+1} :: loss: {loss_tensor.item()}')

#         torch.cuda.synchronize()
#         if rank==0:
#             ## compute metrics
#             train_f1_score = f1_score(actuals, preds, average='macro')
#             train_loss = epoch_losses/len(train_dataloader)
#             logger.info(f'epoch:{epoch}:: train_loss:{train_loss} | train_f1_score: {train_f1_score}')
#             train_losses.append((epoch,train_loss))
#             train_f1_scores.append((epoch, train_f1_score))

#             ## validation step
#             if config['training']['validation'] and epoch%config['training']['val_frequency']==0:
#                 out, test_loss, test_f1_score = testModel(test_dataloader, model, rank, criteria, config)
#                 val_losses.append((epoch, test_loss))
#                 val_f1_scores.append((epoch, test_f1_score))
#                 logger.info(f'             :: val_loss:{test_loss} | val_f1_score:{test_f1_score}')

#             ## save model
#             if config['training']['save_model'] and epoch%config['training']['save_frequency']==0 and epoch!=0:
#                 file = os.path.join(result_dir, 'weights', f'epoch-{epoch}.pt')
#                 torch.save({
#                     'epoch': epoch,
#                     'model': model.module.state_dict(),
#                     'optim': optimizer.state_dict(),
#                     'loss': train_loss,
#                     'f1_score': train_f1_score, 
#                     'expt': result_dir,
#                     'data_names': config['dataset']['data_names'],
#                     'case_range': config['dataset']['case_range']
#                 }, f=file)
#                 logger.info(f'model saved @{file}')

#             ## plots metrics
#             plot_metrics(train_losses, val_losses, save_path=os.path.join(result_dir, 'plots', 'loss.png'), name='Loss')
#             plot_metrics(train_f1_scores, val_f1_scores, save_path=os.path.join(result_dir, 'plots', 'f1_score.png'), name='F1 Score')

#     # torch.cuda.empty_cache()
#     cleanup(logger, rank)
#     return

def prepareDataMultiGPU(config, logger, rank=None, world_size=None):
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
    
    data_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(
                                train_dataset,
                                batch_size=1,
                                sampler=data_sampler,
                                num_workers=config['n_workers'],           # ← disable multiprocessing in dataloader
                                pin_memory=False,        # ← avoid the pin memory crash
                                drop_last=True
                            )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers']) 

    return train_dataloader, test_dataloader, data_sampler  


# Initializes the DDP environment
def ddp_setup(rank, world_size, logger):
    # Set the master node address and port for communication.
    # Useful for multi-node SLURM jobs (though you're using a single node with 2 GPUs).
    os.environ["MASTER_ADDR"] = os.getenv("SLURM_NODELIST").split(',')[0]
    os.environ["MASTER_PORT"] = "12355"  # Consistent port across ranks
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Limit visible GPUs (optional if SLURM handles it)

    try:
        torch.cuda.set_device(rank)  # Assign the specific GPU to this process
        dist.init_process_group("nccl", rank=rank, world_size=world_size)  # Initialize DDP with NCCL backend
    except RuntimeError as e:
        logger.error(f"Failed to initialize process group: {e}")
        raise
    torch.cuda.set_device(rank)  # Reaffirm device assignment (redundant, but safe)

# Spawns multiple processes, each handling one GPU
def init_dpp(world_size, model, config, logger, result_dir):
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    mp.set_start_method("spawn", force=True)  # Set multiprocessing start method (platform-specific)
    mp.spawn(ddp_train,                        # Spawn one process per GPU
            args=(world_size, model, config, logger, result_dir),
            nprocs=world_size,
            join=True)

# Cleans up the process group after training ends
def cleanup(logger, rank):
    try:
        dist.barrier(device_ids=[rank])  # Synchronize before shutdown (optional but recommended)
        dist.destroy_process_group()     # Tear down the distributed setup
    except RuntimeError as e:
        logger.error(f"Failed to destroy process group: {e}")

# Main training function for each process (one per GPU)
def ddp_train(rank, world_size, model, config, logger, result_dir):
    ddp_setup(rank, world_size, logger)  # Set up DDP for this rank
    dist.barrier()  # Optional sync point before training begins

    # Load and distribute dataset across GPUs
    train_dataloader, test_dataloader, sampler = prepareDataMultiGPU(config, logger, rank=rank, world_size=world_size)
    logger.info(f"[Rank {rank}] Using device: {torch.cuda.current_device()}")

    if config['training']['retrain_model']:
        model, optimizer = load_model(model, optimizer, logger)  # Resume training from checkpoint if needed

    if config['training']['freeze_attentions']:
        logger.info(f'Freezing attention layers: {config["training"]["freeze_attention_layers"]}')
        model.freeze_attention_layers(config['training']['freeze_attention_layers'])  # Freeze specific transformer layers

    model.to(rank)  # Move model to the GPU assigned to this rank

    # Wrap the model with DDP for multi-GPU synchronization
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optimizer = get_optimizer(config, model)  # Load optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing']).to(rank)

    # Metric trackers
    train_losses, val_losses = [], []
    train_f1_scores, val_f1_scores = [], []

    logger.info('========== Starting model training. ==========')
    for epoch in range(1, config['training']['epochs']+1):
        # train_dataloader.sampler.set_epoch(epoch)  # For shuffling consistency across workers
        logger.info(f"[Rank {rank}] Starting epoch {epoch}")
        sampler.set_epoch(epoch) # torch.cuda.empty_cache()  # Optional memory cleanup
        epoch_losses = 0
        model.train()
        preds, actuals, i = [], [], 0

        ## -------- TRAINING LOOP -------- ##
        for batch in train_dataloader:
            # Move batch data to GPU
            x = batch.x.to(rank)
            laplacian_pe = batch.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(rank)
            x = torch.cat([x, laplacian_pe], dim=2)  # Combine input with positional encoding

            edge_index = batch.edge_index.to(rank)
            edge_attr = batch.edge_attr.to(rank)
            y = batch.y.to(rank)[0] - 1  # Adjust label indexing

            # Forward pass
            out = model(x, edge_index, edge_attr)

            # Loss computation
            loss = criteria(out, y)

            # All-reduce the loss across processes for logging consistency
            loss_tensor = torch.tensor(loss.item(), device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_losses += loss_tensor.item() / world_size

            # Backward pass + optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Collect predictions only from rank 0 (optional)
            if rank == 0:
                preds.append(int(out.argmax()))
                actuals.append(int(batch.y[0]) - 1)
                i += 1
                logger.debug(f'epoch: {epoch+1}-batch: {i+1} :: loss: {loss_tensor.item()}')

        torch.cuda.synchronize()  # Wait for all kernels to finish before proceeding

        ## -------- METRICS, VALIDATION, SAVING (rank 0 only) -------- ##
        print(rank)
        if rank == 0:
            train_f1_score = f1_score(actuals, preds, average='macro')
            train_loss = epoch_losses / len(train_dataloader)
            logger.info(f'epoch:{epoch}:: train_loss:{train_loss} | train_f1_score: {train_f1_score}')
            train_losses.append((epoch, train_loss))
            train_f1_scores.append((epoch, train_f1_score))

            if config['training']['validation'] and epoch % config['training']['val_frequency'] == 0:
                out, test_loss, test_f1_score = testModel(test_dataloader, model, rank, criteria, config)
                val_losses.append((epoch, test_loss))
                val_f1_scores.append((epoch, test_f1_score))
                logger.info(f'             :: val_loss:{test_loss} | val_f1_score:{test_f1_score}')

            # Save model checkpoint
            if config['training']['save_model'] and epoch % config['training']['save_frequency'] == 0 and epoch != 0:
                file = os.path.join(result_dir, 'weights', f'epoch-{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),  # model.module because of DDP
                    'optim': optimizer.state_dict(),
                    'loss': train_loss,
                    'f1_score': train_f1_score,
                    'expt': result_dir,
                    'data_names': config['dataset']['data_names'],
                    'case_range': config['dataset']['case_range']
                }, f=file)
                logger.info(f'model saved @{file}')

            # Plot and save training curves
            plot_metrics(train_losses, val_losses, save_path=os.path.join(result_dir, 'plots', 'loss.png'), name='Loss')
            plot_metrics(train_f1_scores, val_f1_scores, save_path=os.path.join(result_dir, 'plots', 'f1_score.png'), name='F1 Score')

    cleanup(logger, rank)  # End of training for this process
    return
