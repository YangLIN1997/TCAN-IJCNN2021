import argparse
import logging
import os
import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
import utils
import model.Models as Models_Sanyo
from early_stopping import *
from evaluate import evaluate
from dataloader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# from apex import amp

#
logger = logging.getLogger('Transformer.Train')

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='Solar', help='Name of the dataset') # elect Sanyo solar
# parser.add_argument('--dataset', default='elect', help='Name of the dataset') # elect Sanyo solar
# parser.add_argument('--dataset', default='Sanyo', help='Name of the dataset')  # elect Sanyo solar
parser.add_argument('--dataset', default='Hanergy', help='Name of the dataset') # elect Sanyo solar
# parser.add_argument('--gridsearch', default=None, help='Whether to search hyperparameter')
parser.add_argument('--search-hyperparameter', default=None, help='Whether to search hyperparameter, BO, GS')
parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
# parser.add_argument('--model-name', default='base_model_Solar', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Solar_2dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Solar_3dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Solar_4dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Solar_5dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_elect', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_elect_7d', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo_soft', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo_sparse', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo_2dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo_3dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo_4dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Sanyo_5dh', help='Directory containing params.json')
parser.add_argument('--model-name', default='base_model_Hanergy', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Hanergy_2dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Hanergy_3dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Hanergy_4dh', help='Directory containing params.json')
# parser.add_argument('--model-name', default='base_model_Hanergy_5dh', help='Directory containing params.json')
parser.add_argument('--relative-metrics', default=False, help='Whether to normalize the metrics by label scales')
# parser.add_argument('--relative-metrics', default=True, help='Whether to normalize the metrics by label scales')
parser.add_argument('--sampling', default=True, help='Whether to sample during evaluation')
parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
parser.add_argument('--tqdm', default=False, help='Whether to disable tqdm progress bar')
parser.add_argument('--seed', default=0, help='Set random seed')
# parser.add_argument('--restore-file', default='best',#None,
parser.add_argument('--restore-file', default=None,
                    help='Optional, name of the file in --model_dir containing weights to reload before \
                    training')  # 'best' or 'epoch_#'


def train_Sanyo(model: nn.Module,
                optimizer: optim,
                loss_fn,
                train_loader: DataLoader,
                test_loader: DataLoader,
                params: utils.Params,
                epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        test_loader: load test data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()
    torch.autograd.set_detect_anomaly(True)
    loss_epoch = np.zeros(len(train_loader))


    for i, (train_batch, labels_batch) in enumerate(tqdm(train_loader, disable=args.tqdm)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]

        train_batch = train_batch.to(torch.float32).to(params.device)  # not scaled
        labels_batch = labels_batch.to(torch.float32).to(params.device)  # not scaled

        loss = torch.zeros(1, device=params.device)
        mu,Sigma = model.forward(train_batch[:, 0:params.predict_start, :].clone(),
                           train_batch[:, params.predict_start:, :].clone())
        loss += loss_fn(mu, Sigma, labels_batch, params.predict_start)
        if cuda_exist:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                while scaled_loss>1e5:
                  scaled_loss/=10
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        loss = loss.item() / params.train_window  # loss per timestep
        loss_epoch[i] = loss


    return loss_epoch


def train_and_evaluate_Sanyo(model: nn.Module,
                             train_loader: DataLoader,
                             valid_loader: DataLoader,
                             test_loader: DataLoader,
                             optimizer: optim, scheduler, loss_fn,
                             params: utils.Params,
                             restore_file: str = None) -> None:
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        valid_loader: load valid data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    '''
    start_epoch = 0
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file + '.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        start_epoch = 101
        if restore_file != 'best':
            start_epoch = int(restore_file[6:])+1
        if restore_file == 'best':
            test_metrics = evaluate(model, loss_fn, test_loader, params, start_epoch, sample=args.sampling, plot=True)
            return test_metrics
    logger.info('begin training and evaluation')
    best_test_loss = float('inf')
    best_valid_loss = float('inf')
    train_len = len(train_loader)
    ND_valid_summary = np.zeros(params.num_epochs)
    ND_test_summary = np.zeros(params.num_epochs)
    loss_valid_summary = np.zeros(params.num_epochs)
    loss_train_summary = np.zeros((train_len * params.num_epochs))
    loss_test_summary = np.zeros(params.num_epochs)

    es = EarlyStopping(patience=10)

    # nvidia_smi.nvmlInit()

    for epoch in range(start_epoch, params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        loss_train_summary[epoch * train_len:(epoch + 1) * train_len] = train_Sanyo(model, optimizer, loss_fn,
                                                                                    train_loader,
                                                                                    test_loader, params, epoch)

        # if cuda_exist:
        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        #     # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
        #     res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        #     logger.info(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%, cpu: {psutil.cpu_percent()}')

        logger.info(f'train_loss: {loss_train_summary[epoch * train_len:(epoch + 1) * train_len].mean()}')

        test_metrics = evaluate(model, loss_fn, test_loader, params, epoch, sample=args.sampling, plot=True)

        ND_test_summary[epoch] = test_metrics['ND']
        loss_test_summary[epoch] = test_metrics['test_loss']
        if loss_test_summary[epoch] <= best_test_loss:
            best_test_loss = loss_test_summary[epoch]
            is_best = loss_test_summary[epoch] <= best_test_loss
        else:
            is_best = False

        if params.early_stopping == True:
            valid_metrics = evaluate(model, loss_fn, valid_loader, params, epoch, sample=args.sampling, plot=False)
            loss_valid_summary[epoch] = valid_metrics['test_loss']
            if loss_valid_summary[epoch] <= best_valid_loss:
                best_valid_loss = loss_valid_summary[epoch]
                is_best = loss_valid_summary[epoch] <= best_valid_loss
            else:
                is_best = False
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              epoch=epoch,
                              is_best=is_best,
                              checkpoint=params.model_dir)
        with open(os.path.join(params.model_dir, 'loss.npy'), 'wb') as f:
            np.save(f, loss_train_summary[:(epoch + 1) * train_len])
            np.save(f, loss_test_summary[:(epoch + 1)])
            np.save(f, loss_valid_summary[:(epoch + 1)])
            np.save(f, train_len)
        if is_best:
            logger.info('- Found new best ND')
            best_test_ND = ND_test_summary[epoch]
            best_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best ND is: %.5f' % best_test_ND)

        utils.plot_all_epoch(ND_test_summary[:epoch + 1], args.dataset + '_test_ND', params.plot_dir)
        utils.plot_all_epoch(loss_train_summary[:(epoch + 1) * train_len], args.dataset + '_train_loss',
                             params.plot_dir)
        utils.plot_all_epoch(loss_test_summary[:(epoch + 1)], args.dataset + '_test_loss', params.plot_dir)
        utils.plot_all_epoch(loss_valid_summary[:(epoch + 1)], args.dataset + '_valid_loss', params.plot_dir)

        last_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

        if params.early_stopping == True and es.step(
                torch.from_numpy(np.array(loss_valid_summary[epoch])).to(torch.float32).to(params.device), epoch):
            logger.info('Best epoch is: %d' % es.best_epoch)
            break  # early stop criterion is met, we can stop now

        # optimizer.param_groups[0]['lr'] = es.step_lr(optimizer.param_groups[0]['lr'], 0.5, 2)

    if args.save_best:
        f = open('./param_search.txt', 'w')
        f.write('-----------\n')
        list_of_params = args.search_params.split(',')
        print_params = ''
        for param in list_of_params:
            param_value = getattr(params, param)
            print_params += f'{param}: {param_value:.2f}'
        print_params = print_params[:-1]
        f.write(print_params + '\n')
        f.write('Best ND: ' + str(best_test_ND) + '\n')
        logger.info(print_params)
        logger.info(f'Best ND: {best_test_ND}')
        f.close()
        utils.plot_all_epoch(ND_test_summary, print_params + '_test_ND', location=params.plot_dir)
        utils.plot_all_epoch(loss_test_summary, print_params + '_test_loss', location=params.plot_dir)

    return best_test_ND


def main():
    global args, cuda_exist
    # Load the parameters from json file
    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)

    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    params = utils.Params(json_path)

    if args.search_hyperparameter is not None:
        # print('searching!')
        logging.disable(logging.CRITICAL)
        para_json_path = os.path.join(os.path.dirname(os.path.dirname(json_path)), 'params.json')
        ND_json_path = os.path.join(os.path.dirname(os.path.dirname(json_path)), 'ND_best.json')
        para_base_model_json_path = os.path.join(os.path.join('experiments', 'base_model'), 'params.json')
        best_params = utils.Params(json_path)

    params.relative_metrics = args.relative_metrics
    params.sampling = args.sampling
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    seed = args.seed

    # use GPU if available
    # cuda_exist = torch.cuda.is_available()
    # cuda_exist = False
    # Set random seeds for reproducible experiments if necessary
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    utils.set_logger(os.path.join(model_dir, 'train.log'))

    if cuda_exist:
        # print('Using Cuda...')
        params.device = torch.device('cuda')
        logger.info('Using Cuda...')
        logger.info(torch.cuda.get_device_name(0))
        model = Models_Sanyo.Transformer(params,args.dataset).cuda()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        params.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = Models_Sanyo.Transformer(params,args.dataset)
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print(pytorch_total_params)
    # torch.set_num_threads(4)
    logger.info('Loading the datasets...')
    # print(model)

    valid_loader = None
    if args.search_hyperparameter is not None:
        train_set = TrainDataset(data_dir, 'search_' + args.dataset)
        if params.early_stopping == True:
            valid_set = ValidDataset(data_dir, 'search_' + args.dataset)
            valid_loader = DataLoader(valid_set, batch_size=params.predict_batch, sampler=RandomSampler(valid_set),
                                      num_workers=4, worker_init_fn=np.random.seed(12))
        test_set = TestDataset(data_dir, 'search_' + args.dataset)
    else:
        train_set = TrainDataset(data_dir, args.dataset)
        if params.early_stopping == True:
            valid_set = ValidDataset(data_dir, args.dataset)
            valid_loader = DataLoader(valid_set, batch_size=params.predict_batch, sampler=RandomSampler(valid_set),
                                      num_workers=4, worker_init_fn=np.random.seed(12))
        test_set = TestDataset(data_dir, args.dataset)
    # sampler = WeightedSampler(data_dir, args.dataset) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=RandomSampler(train_set), num_workers=4,
                              worker_init_fn=np.random.seed(12))
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4,
                             worker_init_fn=np.random.seed(12))

    logger.info('Loading complete.')
    # logger.info(f'Model: \n{str(model)}')

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    if cuda_exist:
        model, optimizer = amp.initialize(model, optimizer,
                                          loss_scale=128,
                                          verbosity=0, opt_level="O1",
                                          min_loss_scale=None,
                                          max_loss_scale=2.**10)

    # fetch loss function
    loss_fn = Models_Sanyo.loss_fn

    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    ND = train_and_evaluate_Sanyo(model=model,
                                  train_loader=train_loader,
                                  valid_loader=valid_loader,
                                  test_loader=test_loader,
                                  optimizer=optimizer,
                                  scheduler=None,
                                  loss_fn=loss_fn,
                                  params=params,
                                  restore_file=args.restore_file)

    if args.search_hyperparameter is not None:
        best_params.save_best(para_json_path, para_base_model_json_path, ND_json_path, ND)

    return ND


if __name__ == '__main__':
    main()