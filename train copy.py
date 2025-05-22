import os
import torch
import yaml
import pytz
import argparse
from datetime import datetime
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from torch_geometric.loader import DataLoader

from layers import GraphTransformer
from data import extractData, transformData
from utils import *

# ref: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py

def prepareData(config):
    dataset = extractData(data_path=config['dataset']['data_path'],
                          data_names=config['dataset']['data_names'],
                          window_size=config['dataset']['window_size'],
                          n_nodes=config['dataset']['n_nodes'],
                          n_edges=config['dataset']['n_edges'],
                          n_edge_features=config['dataset']['n_edge_features'],
                          n_node_features=config['dataset']['n_node_features'],
                          case_range=config['dataset']['case_range'], 
                          stride=config['dataset']['stride'],
                          data_portion=config['dataset']['data_portion_from_end'])
    
    dataset = transformData(k=config['dataset']['k'], dataset=dataset)
    
    train_dataset, test_dataset = train_test_split(dataset, test_size=config['dataset']['test_ratio'], random_state=config['random_seed'])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config['n_workers']) 

    return train_dataloader, test_dataloader  


def trainModel(config, train_dataloader, test_dataloader, model, device, logger, result_dir):
    model = model.to(device)    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], eps=1e-9)

    if config['training']['l2_regularize']:
        params_decay = [p for n,p in model.named_parameters() if "bias" not in n]
        params_no_decay = [p for n,p in model.named_parameters() if "bias" in n]

        optimizer = torch.optim.AdamW(
            [{"params": params_decay, "weight_decay": config['training']['l2_lambda']},
            {"params": params_no_decay, "weight_decay": 0}],
            lr=config['training']['learning_rate']
        )

    criteria = torch.nn.CrossEntropyLoss(label_smoothing=config['training']['label_smoothing']).to(device)

    ## load model
    if config['training']['retrain_model']:
        # checkpoint_path = os.path.join(os.getcwd(), config['training']['pretrained_model_path'])
        checkpoint_path = os.path.join(config['result_dir'], config['training']['pretrained_model_path'])
        assert os.path.exists(checkpoint_path), 'Provided pretrained model does not exists.'

        checkpoint = torch.load(checkpoint_path)
        logger.info('')
        logger.info(f"Loading pretrained model from: {config['training']['pretrained_model_path']}")
        logger.info(f"> Used data info:: names: {checkpoint['data_names']}, range: {checkpoint['case_range']}")
        logger.info(f"> epoch:{checkpoint['epoch']}:: train_loss:{checkpoint['loss']} | train_f1_score: {checkpoint['f1_score']}")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])

    logger.info('')
    if config['training']['freeze_attentions']:
        logger.info(f'Freezing attention layers: {config["training"]["freeze_attention_layers"]}')
        model.freeze_attention_layers(config['training']['freeze_attention_layers'])

    train_losses, val_losses = [], []
    train_f1_scores, val_f1_scores = [], []
    logger.info('========== Starting model training. ==========')
    for epoch in range(config['training']['epochs']):
        torch.cuda.empty_cache()
        epoch_losses = 0
        model.train()        
        preds, actuals, i = [], [], 0
        
        ## train model 
        for batch in tqdm(train_dataloader, desc=f'epoch-{epoch}:: train'):
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
            # preds.append(out.cpu().argmax())
            # actuals.append(int(batch.y.cpu()[0]-1))
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
            out, test_loss, test_f1_score = testModel(test_dataloader, model, device, criteria)
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

def testModel(test_dataloader, model, device, criteria):
    model.eval()    
    data = next(iter(test_dataloader)).to(device)
    pe = data.laplacian_eigenvector_pe.unsqueeze(1).expand(-1, config['dataset']['window_size'], -1).to(device)
    out = model(torch.cat([data.x, pe], dim=2), data.edge_index, data.edge_attr).cpu()
    data = data.cpu()   
    print('\nSample test') 
    print('> model output', out.detach().numpy())
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

def runProcess(config):
    w = Watch()
    w.start()
    ## prepare experiment directory
    now = datetime.now(pytz.timezone('Asia/Kolkata'))
    tm = now.strftime('%Y-%m-%d %H:%M')
    # results = os.path.join(os.getcwd(), 'results')
    results = os.path.join(config['result_dir'], 'results')
    os.makedirs(results, exist_ok=True)
    entries = os.listdir(results)
    
    expt = "debug" if config['test_mode'] else get_max_expt_number(entries)+1
    result_dir = os.path.join(results, f'expt-{expt}| {tm}') if not config['test_mode'] else os.path.join(config['result_dir'], 'debug')
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'plots'), exist_ok=True)

    copy_file(os.path.join(os.getcwd(), args.config_file), os.path.join(result_dir, 'config.yml'))
    copy_file(os.path.join(os.getcwd(), 'layers.py'), os.path.join(result_dir, 'layers.py'))
    copy_file(os.path.join(os.getcwd(), 'train.py'), os.path.join(result_dir, 'train.py'))

    ## prepare logging setup
    logging = logging_setup()
    logger = get_logger(f'expt-{expt}', result_dir)

    logger.info('')
    logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(f'++++++++++++++++++++++++++ Experiment: {expt} ++++++++++++++++++++++++++')
    logger.info(f'Description: {config['desc']}')

    ## training process 
    logger.info('')
    logger.info('Loading dataset.')
    logger.info(f"> Data names: {config['dataset']['data_names']}")
    logger.info(f"> Case range: {config['dataset']['case_range']}")
    train_dataloader, test_dataloader = prepareData(config)

    logger.info('')
    logger.info('Loading model.')
    ## load model
    model = GraphTransformer(d_model=config['model']['d_model'],
                             num_nodes=config['dataset']['n_nodes'],
                             num_heads=config['model']['n_heads'],
                             node_features=config['dataset']['n_node_features']+config['dataset']['k'],
                             edge_features=config['dataset']['n_edge_features'],
                             dropout=config['model']['dropout'],
                             use_bias=config['model']['use_bias'],
                             num_layers=config['model']['n_layers'], 
                             num_fault_types=config['dataset']['num_fault_types'])
    
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

    # heads = [4, 8, 16, 32, 64, 128]
    # n_layers = [6, 7]
    # drops = [0.3, 0.4, 0.5, 0.7]
    # desc_format = config['desc']

    # for i, nlayers in enumerate(n_layers):
    #     config['model']['n_layers'] = nlayers
    #     config['desc'] = desc_format.format(nlayers)

    # for i, drop in enumerate(drops):
    #     config['model']['dropout'] = drop
    #     config['desc'] = desc_format.format(drop)

    # for i, head in enumerate(heads):
    #     config['model']['n_heads'] = head
    #     config['desc'] = desc_format.format(head)

    #     runProcess(config)




