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
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.optim.lr_scheduler import StepLR
# DistributedDataParallel: Use single-machine multi-GPU 
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed import init_process_group, destroy_process_group

# from layers import GraphTransformer, PowerFormer
# from data import extractData, transformData
from utils import *

# def testModel(test_dataloader, model, device, criteria, config):
#     model.eval()    
#     data = next(iter(test_dataloader)).to(device)
#     pe = data.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(device)
#     out = model(torch.cat([data.x, pe], dim=2), data.edge_index, data.edge_attr).cpu()
#     data = data.cpu()   
#     print('\nSample test') 
#     prob = softmax(out.detach(), dim=0, dtype=torch.float32).numpy().tolist()
#     trimmed_prob = [float(f"{num:.5f}") for num in prob]
#     print('> model output', trimmed_prob)
#     print(f'> predicted class: {out.argmax()}, actual class: {data.y[0]-1}')

#     test_loss = 0
#     preds, actuals = [], []
    
#     ## test model
#     for batch in test_dataloader:
#         x = batch.x.to(device)
#         laplacian_pe = batch.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(device)
#         x = torch.cat([x, laplacian_pe], dim=2) # shape: (n_nodes, time, node_features+PE)
#         edge_index = batch.edge_index.to(device)
#         edge_attr = batch.edge_attr.to(device)
        
#         out = model(x, edge_index, edge_attr)
#         y = batch.y.to(device)[0]-1 
#         loss = criteria(out, y)
#         test_loss += loss.item()

#         preds.append(int(out.argmax()))
#         actuals.append(int(batch.y[0])-1)

#     return out, test_loss/len(test_dataloader), f1_score(actuals, preds, average='macro')

# def prepareData(config, logger, rank=None, world_size=None):
#     dataset = extractData(data_path=config['dataset']['data_path'],
#                           data_names=config['dataset']['data_names'],
#                           window_size=config['dataset']['window_size'],
#                           n_nodes=config['dataset']['n_nodes'],
#                           n_edges=config['dataset']['n_edges'],
#                           edge_features=config['dataset']['edge_features'],
#                           node_features=config['dataset']['node_features'],
#                           case_range=config['dataset']['case_range'], 
#                           stride=config['dataset']['stride'],
#                           data_portion=config['dataset']['data_portion_from_end'],
#                           ignored_fault_locations=config['dataset']['ignored_fault_locations'],
#                           task=config['task'],
#                           current_as_node_features=config['dataset']['current_as_node_features'])
    
#     dataset = transformData(k=config['dataset']['k'], dataset=dataset)
#     logger.info(f'Dataset Size: {len(dataset)}')
#     train_dataset, test_dataset = train_test_split(dataset, test_size=config['dataset']['test_ratio'], random_state=config['random_seed'])
    
#     if config['training']['multi_gpu']:
#         train_dataloader = DataLoader(train_dataset, 
#                                       batch_size=1, 
#                                       num_workers=config['n_workers'], 
#                                       sampler=DistributedSampler(train_dataset, shuffle=True, rank=rank, num_replicas=world_size, drop_last=True))
#     else:    
#         train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers'])

#     test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers']) 

#     return train_dataloader, test_dataloader  

# def get_optimizer(config, model):
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], eps=1e-9)

#     if config['training']['l2_regularize']:
#         params_decay = [p for n,p in model.named_parameters() if "bias" not in n]
#         params_no_decay = [p for n,p in model.named_parameters() if "bias" in n]

#         optimizer = torch.optim.AdamW(
#             [{"params": params_decay, "weight_decay": config['training']['l2_lambda']},
#             {"params": params_no_decay, "weight_decay": 0}],
#             lr=config['training']['learning_rate']
#         )

#     return optimizer

# def load_model(model, optimizer, logger, config):
#     checkpoint_path = os.path.join(config['result_dir'], config['training']['pretrained_model_path'])
#     assert os.path.exists(checkpoint_path), 'Provided pretrained model does not exists.'

#     checkpoint = torch.load(checkpoint_path)
#     logger.info('')
#     logger.info(f"Loading pretrained model from: {config['training']['pretrained_model_path']}")
#     logger.info(f"> Used data info:: names: {checkpoint['data_names']}, range: {checkpoint['case_range']}")
#     logger.info(f"> epoch:{checkpoint['epoch']}:: train_loss:{checkpoint['loss']} | train_f1_score: {checkpoint['f1_score']}")

#     if config['training']['only_load_attention']:
#         model_state = model.state_dict()
#         filtered_dict = {}
#         for k, v in checkpoint['model'].items():
#             if k in model_state and v.shape == model_state[k].shape and ('layers' in k):
#                 filtered_dict[k] = v
#         model_state.update(filtered_dict)
#         model.load_state_dict(model_state)
#     else:
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optim'])

#     return model, optimizer

def trainModel(config, train_dataloader, test_dataloader, model, device, logger, result_dir):
    model = model.to(device)    
    optimizer = get_optimizer(config, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing']).to(device)

    ## load model
    if config['training']['retrain_model']:
        model, optimizer = load_model(model, optimizer, logger, config)

    logger.info('')
    if config['training']['freeze_attentions']:
        logger.info(f'Freezing attention layers: {config["training"]["freeze_attention_layers"]}')
        model.freeze_attention_layers(config['training']['freeze_attention_layers'])

    train_losses, val_losses = [], []
    train_f1_scores, val_f1_scores = [], []
    logger.info('========== Starting model training. ==========')
    for epoch in range(1, config['training']['epochs']+1):
        torch.cuda.empty_cache()
        epoch_losses = 0
        model.train()        
        preds, actuals, i = [], [], 0
        
        ## train model 
        for batch in train_dataloader:
            x = batch.x.to(device) # shape: (n_nodes, time, node_features)
            laplacian_pe = batch.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(device)
            x = torch.cat([x, laplacian_pe], dim=2) # shape: (n_nodes, time, node_features+PE)
            #.view(config['dataset']['n_nodes'], config['dataset']['window_size'], -1) 
            edge_index = batch.edge_index.to(device) # shape: (2, n_edges)
            edge_attr = batch.edge_attr.to(device) # shape: (n_edges, time, edge_features)
            y = batch.y.to(device)[0]-1 

            out = model(x, edge_index, edge_attr) # shape: (num_fault_types,)

            loss = criteria(out, y)
            epoch_losses += loss.item()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            preds.append(int(out.argmax()))
            actuals.append(int(batch.y[0])-1)
            i += 1
            logger.debug(f'epoch: {epoch+1}-batch: {i+1} :: loss: {loss.item()}')

        ## compute metrics
        train_f1_score = f1_score(actuals, preds, average='macro')
        train_loss = epoch_losses/len(train_dataloader)
        logger.info(f'epoch:{epoch}:: train_loss:{train_loss} | train_f1_score: {train_f1_score}')
        train_losses.append((epoch,train_loss))
        train_f1_scores.append((epoch, train_f1_score))

        ## validation step
        if config['training']['validation'] and epoch%config['training']['val_frequency']==0:
            out, test_loss, test_f1_score = testModel(test_dataloader, model, device, criteria, config)
            val_losses.append((epoch, test_loss))
            val_f1_scores.append((epoch, test_f1_score))
            logger.info(f'             :: val_loss:{test_loss} | val_f1_score:{test_f1_score}')

        ## save model
        if config['training']['save_model'] and epoch%config['training']['save_frequency']==0 and epoch!=0:
            file = os.path.join(result_dir, 'weights', f'epoch-{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
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

    torch.cuda.empty_cache()
    return
