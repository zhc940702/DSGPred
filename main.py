import os
import random

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
from torch_geometric import nn
from network import ISGDRP
from EarlyStopping import *
sys.path.append("..")
import torch.multiprocessing
import argparse
import fitlog
from utils import *
import torch.utils.data
def main():

    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default =200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.0003,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--droprate', type = float, default = 0.3,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--batch_size', type = int, default =64,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default =64,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--rawpath', type=str, default='data/',
                        metavar='STRING', help='rawpath')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for earlystopping (default: 10)')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test')
    parser.add_argument('--weight_path', type=str, default='best2',
                        help='filepath for pretrained weights')
    parser.add_argument('--layer_drug', type=int, default=3,
                        help='filepath for pretrained weights')
    args = parser.parse_args()

    print('model_best1+drop_out: '+ str(args.droprate))
    print('-------------------- Hyperparams --------------------')
    print('weight decay: ' + str(args.weight_decay))
    print('dropout rate: ' + str(args.droprate))
    print('learning rate: ' + str(args.lr))
    print('dimension of embedding: ' + str(args.embed_dim))

    setup_seed(20)
    train_test_model(args)

    print('model_best1+drop_out: ' + str(args.droprate))


def setup_seed(seed):
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  #
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_test_model(args):
    train_loader, test_loader, val_loader = load_data(args)
    all_train, all_val, all_test = [], [], []
    for i in range(5):
        print('*' * 20 + 'Fold {}'.format(i+1) + ' start' + '*' * 20)
        model = ISGDRP(args).to(args.device)
        if args.mode == 'train':
            Regression_criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            log_folder = os.path.join(os.getcwd(), "result/log", model._get_name())
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            fitlog.set_log_dir(log_folder)
            fitlog.add_hyper(args)
            fitlog.add_hyper_in_file(__file__)

            stopper = EarlyStopping(mode='lower', patience=args.patience)
            for epoch in range(1, args.epochs + 1):
                print("=====Epoch {}".format(epoch))
                print("Training...")
                train_loss = train(model,train_loader[i], Regression_criterion, optimizer, args)
                fitlog.add_loss(train_loss.item(), name='Train MSE', step=epoch)


                print('Evaluating...')
                rmse, mae, _, _ = validate(model, val_loader[i],args)
                print("Validation rmse:{}".format(rmse))
                fitlog.add_metric({'val': {'RMSE': rmse}}, step=epoch)

                early_stop = stopper.step(rmse+mae, model)
                if early_stop:
                    break

            print('EarlyStopping! Finish training!')
            print('Testing...')
            stopper.load_checkpoint(model)
            torch.save(model.state_dict(), 'weight/{}_fold{}.pth'.format(args.weight_path, i))
            train_rmse, train_MAE, train_r2, train_r = validate(model, train_loader[i], args)
            val_rmse, val_MAE, val_r2, val_r = validate(model, val_loader[i], args)
            test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader[i], args)
            print('Train reslut: rmse:{} mae:{} r:{}'.format(train_rmse,train_MAE, train_r))
            print('Val reslut: rmse:{} mae:{} r:{}'.format(val_rmse, val_MAE, val_r))
            print('Test reslut: rmse:{} mae:{}  r:{}'.format(test_rmse, test_MAE, test_r))

            fitlog.add_best_metric(
                {'epoch': epoch - args.patience,
                    "train": {'RMSE': train_rmse, 'MAE': train_MAE, 'pearson': train_r, "R2": train_r2},
                    "valid": {'RMSE': stopper.best_score, 'MAE': val_MAE, 'pearson': val_r, 'R2': val_r2},
                    "test": {'RMSE': test_rmse, 'MAE': test_MAE, 'pearson': test_r, 'R2': test_r2}})

            all_train.append([train_rmse.cpu(), train_MAE, train_r])
            all_val.append([val_rmse.cpu(), val_MAE, val_r])
            all_test.append([test_rmse.cpu(), test_MAE, test_r])
        elif args.mode == 'test':

            model.load_state_dict(
                torch.load('weight/{}_fold{}.pth'.format(args.weight_path, i), map_location=args.device)['model_state_dict'])
            test_rmse, test_MAE, test_r2, test_r = validate(model, test_loader, args.device)
            print('Test RMSE: {}, MAE: {}, R2: {}, R: {}'.format(round(test_rmse.item(), 4), round(test_MAE, 4),
                                                             round(test_r2, 4), round(test_r, 4)))
        print('*' * 20 + 'Fold {}'.format(i+1) + ' end' + '*' * 20)

    all_train = np.array(all_train)
    all_val = np.array(all_val)
    all_test = np.array(all_test)
    print('Train reslut: rmse:{:.4f}±{:.4f} mae:{:.4f}±{:.4f} r:{:.4f}±{:.4f}'.format(
    np.mean(all_train, axis=0)[0], np.std(all_train, axis=0)[0], np.mean(all_train, axis=0)[1],
        np.std(all_train, axis=0)[1], np.mean(all_train, axis=0)[2], np.std(all_train, axis=0)[2]))
    print('Val reslut: rmse:{:.4f}±{:.4f} mae:{:.4f}±{:.4f} r:{:.4f}±{:.4f}'.format(
        np.mean(all_val, axis=0)[0], np.std(all_val, axis=0)[0], np.mean(all_val, axis=0)[1],
        np.std(all_val, axis=0)[1], np.mean(all_val, axis=0)[2], np.std(all_val, axis=0)[2]))
    print('Test reslut: rmse:{:.4f}±{:.4f} mae:{:.4f}±{:.4f} r:{:.4f}±{:.4f}'.format(
        np.mean(all_test, axis=0)[0], np.std(all_test, axis=0)[0], np.mean(all_test, axis=0)[1],
        np.std(all_test, axis=0)[1], np.mean(all_test, axis=0)[2], np.std(all_test, axis=0)[2]))

if __name__ == '__main__':
    try:

        main()


    except Exception as exception:
        raise