from sklearn.model_selection import RepeatedKFold
from dhg import Graph, Hypergraph
from models import HGNN
from util.common_utils import *
import copy
import os
import torch
import torch.optim as optim
from config import get_config

# load configuration
cfg = get_config('config/config_WPDP.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set random seed , The random seed can be adjusted according to the experimental requirements

seed = 321
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def HG_supervised_embedding(X, y, train_index, test_index, G):
    seed = 321
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # transform data to device
    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).squeeze().long().to(device)
    train_index = torch.Tensor(train_index).long().to(device)
    test_index = torch.Tensor(test_index).long().to(device)
    G = G.to(device)

    # model initialization
    HGNN_model = HGNN(in_ch=X.shape[1], n_class=2, n_hid=cfg['n_hid'], dropout=cfg['drop_out'])

    HGNN_model = HGNN_model.to(device)

    optimizer = optim.Adam(HGNN_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    # model training
    since = time.time()
    num_epochs = cfg['max_epoch']
    print_freq = cfg['print_freq']
    best_model_wts = copy.deepcopy(HGNN_model.state_dict())
    best_f1 = 0.0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                HGNN_model.train()  # Set model to training mode
            else:
                HGNN_model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            index = train_index if phase == 'train' else test_index

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs, _ = HGNN_model(X, G)
                loss = criterion(outputs[index], y[index])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * X.size(0)
            running_corrects += torch.sum(preds[index] == y.data[index])

            epoch_loss = running_loss / len(index)
            # epoch_acc = running_corrects.double() / len(index)
            epoch_f1 = metrics.f1_score(y.data[index].cpu().detach().numpy(), preds[index].cpu().detach().numpy())

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(HGNN_model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val F1: {best_f1:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val F1: {best_f1:4f}')

    # return result
    _, X_embedding = HGNN_model(X, G)

    return X_embedding[train_index].cpu().detach().numpy(), X_embedding[test_index].cpu().detach().numpy()


# train and test
def train_and_test(project, mode):
    mcc_list = []
    auc_list = []
    F1_list = []
    precision_list = []
    recall_list = []

    X, y, G = predata_and_G(project, mode, k=cfg['K_neigs'])

    # k-fold cross-validation
    kf = RepeatedKFold(n_splits=cfg['n_splits'], n_repeats=cfg['n_repeats'])  # We can modify n_repeats when debugging.
    for train_index, test_index in kf.split(X, y):

        # hypergraph mapping
        if mode.find('HGNN') != -1:
            X_train, X_test = HG_supervised_embedding(X, y, train_index, test_index, G)
        else:
            X_train, X_test = X[train_index], X[test_index]

        # defect prediction
        y_train, y_test = y[train_index], y[test_index]
        precision, recall, fmeasure, auc, mcc = run_evaluation(X_train, y_train, X_test, y_test, cfg)
        mcc_list.append(mcc)
        auc_list.append(auc)
        F1_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)

    avg = []
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(F1_list))
    avg.append(average_value(auc_list))
    avg.append(average_value(mcc_list))

    name = ['precision', 'recall', 'F1', 'auc', 'mcc']
    results = []
    results.append(precision_list)
    results.append(recall_list)
    results.append(F1_list)
    results.append(auc_list)
    results.append(mcc_list)
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)

    # If the folder does not exist, create the folder
    save_path = './results_WPDP/' + project
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # recording mode parameters
    param_suffix = ""
    if mode.find('HGNN') != -1:
        param_suffix = '_' + str(cfg['K_neigs']) + '_' + str(cfg['n_hid']) + '_' + str(cfg['lr']) + '_' + str(
            cfg['drop_out']) + '_' + str(cfg['max_epoch'])

    df.to_csv(save_path + '/' + mode + param_suffix + '.csv')



# Execute  modes
# modes = ['origin', 'metric', 'vector', 'origin_metric', 'origin_vector', 'metric_vector', 'origin_metric_vector',
#          'origin_HGNN', 'metric_HGNN', 'vector_HGNN', 'origin_metric_HGNN', 'origin_vector_HGNN', 'metric_vector_HGNN', 'origin_metric_vector_HGNN']
modes = ['origin_metric_vector_HGNN ']


# Execute  projects
dict_file = open('Subject_WPDP.csv', 'r') #Subject_CPDP.csv  file can be edited to set up  WPDP tasks
lines = dict_file.readlines()
projects = []
for each_line in lines:
    records = each_line.strip().split(',')
    subject = records[0]
    projects.append(subject)

# param opt
opt_K_neigs = [5]
opt_n_hid = [64]
opt_lr = [0.001]
opt_drop_out = [0]
opt_max_epoch = [200]

import itertools

for params_i in itertools.product(opt_K_neigs, opt_n_hid, opt_lr, opt_drop_out, opt_max_epoch):
    cfg['K_neigs'] = params_i[0]
    cfg['n_hid'] = params_i[1]
    cfg['lr'] = params_i[2]
    cfg['drop_out'] = params_i[3]
    cfg['max_epoch'] = params_i[4]

    for project in projects:
        print(project + " Start!")
        for mode in modes:


            train_and_test(project, mode)
