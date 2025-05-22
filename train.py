import os
import torch
from utils import *

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
