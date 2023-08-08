import os
import torch.optim as optim
from util.common_utils import *
from config import get_config
from dhg.models import HGNN
from models import GRL
import itertools
import torch.nn.functional as F
from torch import nn
import random

# load configuration
cfg = get_config('config/config_CVDP.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Embedding according to the full label of the project is used for cross-version defect prediction


def train_and_test(source_project, target_project, mode):
    # Generate hypergraph G of the source project
    seed = 321
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # Generate hypergraph G of the source project
    X_source, y_source, G_source = predata_and_G(source_project, mode, k=cfg['K_neigs'])
    X_source = torch.Tensor(X_source).to(device)
    y_source = torch.Tensor(y_source).squeeze().long().to(device)
    G_source.to(device)
    X_target, y_target, G_target = predata_and_G(target_project, mode, k=cfg['K_neigs'])
    X_target = torch.Tensor(X_target).to(device)
    y_target = torch.Tensor(y_target).squeeze().long().to(device)
    G_target.to(device)
    if mode.find('_HGNN') != -1:

        # model initialization

        print('hello')
        HGNN_model = HGNN(in_channels=X_source.shape[1], num_classes=cfg['encoder'], hid_channels=cfg['n_hid'],
                          drop_rate=cfg['drop_out'])


        cls_model = nn.Sequential(
            nn.Linear(cfg['encoder'], 2),
        ).to(device)


        domain_model = nn.Sequential(
            GRL(),
            nn.Linear(cfg['encoder'], 40),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(40, 2),
        ).to(device)

        models = [HGNN_model, cls_model, domain_model]
        for model in models:
            model.to(device)

        optimizer = optim.Adam(HGNN_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
        criterion = torch.nn.CrossEntropyLoss()

        # model training
        since = time.time()
        num_epochs = cfg['max_epoch']
        print_freq = cfg['print_freq']
        for epoch in range(num_epochs):
            if epoch % print_freq == 0:
                print('-' * 10)
                print(f'Epoch {epoch}/{num_epochs - 1}')

            # Each epoch has a training and validation phas
            scheduler.step()
            for model in models:
                model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            optimizer.zero_grad()
            global rate
            rate = min((epoch + 1) / num_epochs, 0.05)
            with torch.set_grad_enabled(True):
                encoded_source = HGNN_model(X_source, G_source)
                encoded_target = HGNN_model(X_target, G_target)

                # use source classification loss:
                source_logits = cls_model(encoded_source)
                loss = criterion(source_logits, y_source)
                _, preds = torch.max(source_logits, 1)

                # use domain adaptation loss:
                source_domain_preds = domain_model(encoded_source)
                target_domain_preds = domain_model(encoded_target)
                source_domain_cls_loss = criterion(
                    source_domain_preds,
                    torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
                )
                target_domain_cls_loss = criterion(
                    target_domain_preds,
                    torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
                )
                loss_grl = source_domain_cls_loss + target_domain_cls_loss
                loss = loss + cfg['R1'] * loss_grl

                #  use target entropy loss:
                target_logits = cls_model(encoded_target)
                target_probs = F.softmax(target_logits, dim=-1)
                target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

                loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

                loss = loss + cfg['R2'] * loss_entropy * (epoch / num_epochs * 0.01)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * X_source.size(0)
            running_corrects += torch.sum(preds == y_source.data)

            epoch_loss = running_loss / len(X_source)
            # epoch_acc = running_corrects.double() / len(index)
            epoch_f1 = metrics.f1_score(y_source.data.cpu().detach().numpy(), preds.cpu().detach().numpy())

            if epoch % print_freq == 0:
                print(f'Train Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

        time_elapsed = time.time() - since
        print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # Generate Emb data
        X_source = HGNN_model(X_source, G_source)
        X_target = HGNN_model(X_target, G_target)
        X_source = X_source.cpu().detach().numpy()
        X_target = X_target.cpu().detach().numpy()
        y_source = y_source.cpu().detach().numpy()
        y_target = y_target.cpu().detach().numpy()

    # Cross-version defect prediction
    precision, recall, fmeasure, auc, mcc = run_evaluation(X_source, y_source, X_target, y_target, cfg)

    name = ['precision', 'recall', 'F1', 'auc', 'mcc']
    results = []
    results.append(precision)
    results.append(recall)
    results.append(fmeasure)
    results.append(auc)
    results.append(mcc)
    df = pd.DataFrame(data=results)
    df.index = name
    param_suffix = '_' + str(cfg['R1']) + '_' + str(cfg['R2']) + '_' + str(cfg['K_neigs']) + '_' + str(
        cfg['encoder']) + '_' + str(cfg['n_hid']) + '_' + str(cfg['lr']) + '_' + str(
        cfg['drop_out']) + '_' + str(cfg['max_epoch'])

    # If the folder does not exist, create the folder
    save_path = './results_CVDP/' + source_project + '_to_' + target_project
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    df.to_csv(save_path + '/' + mode + param_suffix + '.csv')


# Execute  modes
# modes = ['origin', 'metric', 'vector', 'origin_metric', 'origin_vector', 'metric_vector', 'origin_metric_vector',
#          'origin_HGNN', 'metric_HGNN', 'vector_HGNN', 'origin_metric_HGNN', 'origin_vector_HGNN', 'metric_vector_HGNN', 'origin_metric_vector_HGNN']

modes = ['origin_metric_vector_HGNN ']


# Execute  projects
dict_file = open('Subject_CVDP.csv', 'r')  # Subject_CVDP.csv  file can be edited to set up  CVDP tasks
lines = dict_file.readlines()
source_projects = []
target_projects = []
for each_line in lines:
    records = each_line.strip().split(',')
    source_project = records[0]
    target_project = records[1]
    source_projects.append(source_project)
    target_projects.append(target_project)

# param opt
opt_R1 = [1]
opt_R2 = [1]
opt_K_neigs = [120]
opt_n_encoder = [64]
opt_n_hid = [16]
opt_lr = [0.001]
opt_drop_out = [0.5]
opt_max_epoch = [100]

for params_i in itertools.product(opt_R1, opt_R2, opt_K_neigs, opt_n_encoder, opt_n_hid, opt_lr, opt_drop_out,
                                  opt_max_epoch):
    cfg['R1'] = params_i[1]
    cfg['R2'] = params_i[1]
    cfg['K_neigs'] = params_i[2]
    cfg['encoder'] = params_i[3]
    cfg['n_hid'] = params_i[4]
    cfg['lr'] = params_i[5]
    cfg['drop_out'] = params_i[6]
    cfg['max_epoch'] = params_i[7]

    # cross-version defect prediction
    for i, source_project in enumerate(source_projects):
        target_project = target_projects[i]
        for mode in modes:
            print(mode + ' Mode: ' + source_project + ' to ' + target_project + " Start!")
            train_and_test(source_project, target_project, mode)
