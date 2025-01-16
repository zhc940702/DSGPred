import pickle
import pickle
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm
import torch.nn.functional as F


def read_raw_data():
    with open('./data/IC50/ic_170drug_580cell.pkl', 'rb') as f:
        drug_cell = pickle.load(f)
    ##drug
    with open('./data/drug_fp/drug_ECFP.pkl', 'rb') as f:
        drug_ECFP = pickle.load(f)
    with open('./data/drug_fp/drug_ERGFP.pkl', 'rb') as f:
        drug_ERGFP = pickle.load(f)
    with open('./data/drug_fp/drug_ESPFP.pkl', 'rb') as f:
        drug_ESPFP = pickle.load(f)
    with open('./data/drug_fp/drug_PSFP.pkl', 'rb') as f:
        drug_PSFP = pickle.load(f)
    with open('./data/drug_fp/drug_RDKFP.pkl', 'rb') as f:
        drug_RDKFP = pickle.load(f)
    with open('./data/drug_fp/drug_DFP.pkl', 'rb') as f:
        drug_DFP = pickle.load(f)

    with open('./data/drug_graph/drug_data/zxy_drug_disease.pkl', 'rb') as f:
        drug_disease = pickle.load(f)
    with open('./data/drug_graph/drug_data/zxy_drug_ADR.pkl', 'rb') as f:
        drug_ADR = pickle.load(f)
    with open('./data/drug_graph/drug_data/zxy_drug_gene_all.pkl', 'rb') as f:
        drug_target = pickle.load(f)
    with open('./data/drug_graph/drug_data/zxy_drug_miRNA.pkl', 'rb') as f:
        drug_miRNA = pickle.load(f)
    with open('./data/drug_graph/drug_data/zxy_stitch_combined_score.pkl', 'rb') as f:
        drug_drug = pickle.load(f)




    with open('./data/cell/exp_580cell_706gene.pkl', 'rb') as f:
        exp = pickle.load(f)
    with open('./data/cell/cn_580cell_706gene.pkl', 'rb') as f:
        cn = pickle.load(f)
    with open('./data/cell/mu_580cell_706gene.pkl', 'rb') as f:
        mu = pickle.load(f)

    cell_features_matrix = np.hstack((exp,cn,mu,drug_cell.T))

    drug_DFP = np.array(drug_DFP,dtype=float)
    drug_cell = np.array(drug_cell,dtype=float)
    drug_ECFP = np.array(drug_ECFP, dtype=float)
    drug_ERGFP = np.array(drug_ERGFP, dtype=float)
    drug_ESPFP = np.array(drug_ESPFP, dtype=float)
    drug_PSFP = np.array(drug_PSFP, dtype=float)
    drug_RDKFP = np.array(drug_RDKFP, dtype=float)
    drug_disease = np.array(drug_disease, dtype=float)
    drug_ADR = np.array(drug_ADR, dtype=float)
    drug_target = np.array(drug_target, dtype=float)
    drug_miRNA = np.array(drug_miRNA, dtype=float)
    drug_drug = np.array(drug_drug, dtype=float)

    drug_cell = np.concatenate((drug_cell,np.zeros((170,5181 - len(drug_cell[0])))),axis=1)
    drug_ECFP = np.concatenate((drug_ECFP, np.zeros((170, 5181 - len(drug_ECFP[0])))), axis=1)
    drug_ERGFP = np.concatenate((drug_ERGFP, np.zeros((170, 5181 - len(drug_ERGFP[0])))), axis=1)
    drug_ESPFP = np.concatenate((drug_ESPFP, np.zeros((170, 5181 - len(drug_ESPFP[0])))), axis=1)
    drug_PSFP = np.concatenate((drug_PSFP, np.zeros((170, 5181 - len(drug_PSFP[0])))), axis=1)
    drug_RDKFP = np.concatenate((drug_RDKFP, np.zeros((170, 5181 - len(drug_RDKFP[0])))), axis=1)
    drug_disease = np.concatenate((drug_disease, np.zeros((170, 5181 - len(drug_disease[0])))), axis=1)
    drug_ADR = np.concatenate((drug_ADR, np.zeros((170, 5181 - len(drug_ADR[0])))), axis=1)
    drug_target = np.concatenate((drug_target, np.zeros((170, 5181 - len(drug_target[0])))), axis=1)
    drug_miRNA = np.concatenate((drug_miRNA, np.zeros((170, 5181 - len(drug_miRNA[0])))), axis=1)
    drug_drug = np.concatenate((drug_drug, np.zeros((170, 5181 - len(drug_drug[0])))), axis=1)

    drug_DFP = np.concatenate((drug_DFP, np.zeros((170, 5181 - len(drug_DFP[0])))), axis=1)


    drug_multi_graph = np.load('./data/drug_graph.npy', allow_pickle=True).item()
    # 580  1024   315  2586   881  200  5181  4693  822   636  170
    drug_features_matrix = np.hstack((drug_cell, drug_ECFP, drug_ERGFP, drug_ESPFP, drug_PSFP, drug_RDKFP, drug_disease,
                                     drug_ADR, drug_target, drug_miRNA, drug_drug, drug_DFP))


    return drug_features_matrix, drug_multi_graph,cell_features_matrix



class MyDataset(Dataset):
    def __init__(self, drug_feature,drug_graph,cell_feature, IC):
        super(MyDataset, self).__init__()
        self.drug_feature, self.drug_graph = drug_feature, drug_graph
        self.cell_feature = cell_feature
        self.drug_name = IC[:,1]
        self.Cell_line_name = IC[:,0]
        self.value = IC[:,2]

    def __len__(self):
        return len(self.value)
    def __getitem__(self, index):
        return (self.drug_feature[int(self.drug_name[index])],
                self.drug_graph[int(self.drug_name[index])],
                self.cell_feature[int(self.Cell_line_name[index])],
                self.value[index])


def _collate(samples):
    drug_feature,drug_graph, cell_feature, labels = map(list, zip(*samples))
    batched_drug_feature  = torch.stack([torch.tensor(d) for d in drug_feature], 0)
    batched_drug_graph = Batch.from_data_list(drug_graph)
    batched_cell = torch.stack([torch.tensor(cell) for cell in cell_feature], 0)
    return batched_drug_feature,batched_drug_graph, batched_cell, torch.tensor(labels)



def load_data(args):
    with open('./data/IC50/samples_82833.pkl', 'rb') as f:
        final_sample = pickle.load(f)
    drug_feature,drug_graph,cell_feature = read_raw_data()
    train_loader_list, test_loader_list, val_loader_list = [], [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=20)
    for train_index, val_test_index in kfold.split(final_sample):
        train_set = final_sample[train_index]
        test_index, val_index = train_test_split(val_test_index, test_size=0.5, random_state=20)
        test_set = final_sample[test_index]
        val_set = final_sample[val_index]
        Dataset = MyDataset
        train_dataset = Dataset(drug_feature,drug_graph,cell_feature, train_set)
        test_dataset = Dataset(drug_feature,drug_graph,cell_feature, test_set)
        val_dataset = Dataset(drug_feature,drug_graph,cell_feature, val_set)


        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate,
                                  num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
                                 num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
                                num_workers=0)
        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)
        val_loader_list.append(val_loader)
    return train_loader_list, test_loader_list, val_loader_list




def train(model, train_data, criterion, optimizer, args):
    model.train()
    for idx, data in enumerate(tqdm(train_data, desc='Iteration')):
        batched_drug_feature,batched_drug_graph, batched_cell, batched_label = data
        batched_drug_feature = batched_drug_feature.to(args.device)
        batched_drug_graph = batched_drug_graph.to(args.device)
        batched_cell = batched_cell.to(args.device)
        output = model(batched_drug_feature, batched_drug_graph, batched_cell)
        loss = criterion(output, batched_label.float().to(args.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss:{}'.format(loss))
    return loss


def validate(model, val_loader, args):
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader, desc='Iteration')):
            batched_drug_feature, batched_drug_graph, batched_cell, batched_label = data
            batched_drug_feature = batched_drug_feature.to(args.device)
            batched_drug_graph = batched_drug_graph.to(args.device)
            batched_cell = batched_cell.to(args.device)
            output = model(batched_drug_feature, batched_drug_graph, batched_cell)
            total_loss += F.mse_loss(output, batched_label.float().to(args.device), reduction='sum')
            y_true.append(batched_label)
            y_pred.append(output)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        rmse = torch.sqrt(total_loss / len(val_loader.dataset))
        MAE = mean_absolute_error(y_true.cpu(), y_pred.cpu())
        r2 = r2_score(y_true.cpu(), y_pred.cpu())
        r = pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]

    return rmse, MAE, r2, r
