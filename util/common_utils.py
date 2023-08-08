import numpy as np
import pandas as pd
import time
import scipy.stats as stats
from util.classifier_utils import *
from imblearn.combine import SMOTETomek

import torch
from dhg import Graph, Hypergraph


def run_evaluation(train_X, train_y, test_X, test_y, cfg):
    t = time.time()
    # If the proportion of positive samples in the training set exceeds 40%, it will be balanced
    if (label_sum(train_y) > (int(len(train_y) * 0.4))):
        print("The training data does not need balance.")
        X_resampled, y_resampled = train_X, train_y
    else:
        # data sample
        X_resampled, y_resampled = SMOTETomek().fit_resample(train_X, train_y)
        # shuffle the data and labels
        state = np.random.get_state()
        np.random.shuffle(X_resampled)
        np.random.set_state(state)
        np.random.shuffle(y_resampled)

    # training classifier
    predprob_auc, predprob, precision, recall, fmeasure, auc, mcc = \
        classifier_output('LogisticRegression', X_resampled, y_resampled, test_X, test_y,
                          grid_sear=cfg['grid_sear'])  # False is only for debugging.

    print("precision=", "{:.5f}".format(precision),
          "recall=", "{:.5f}".format(recall),
          "f-measure=", "{:.5f}".format(fmeasure),
          "auc=", "{:.5f}".format(auc),
          "mcc=", "{:.5f}".format(mcc),
          "time=", "{:.5f}".format(time.time() - t))
    return precision, recall, fmeasure, auc, mcc


def label_sum(label_train):
    label_sum = 0
    for each in label_train:
        label_sum = label_sum + each
    return label_sum


def average_value(list):
    return float(sum(list)) / len(list)


def extract_project_feature(file_path):
    "Extract the feature in the project"

    if file_path.find('.csv') != -1:
        data = pd.read_csv(file_path, header=0, index_col=False)
    elif file_path.find('.emb') != -1:
        data = pd.read_csv(file_path, header=None, sep=" ", index_col=False)
    else:
        raise Exception('There is no way to handle files in formats other than csv and emb')
    if 'Metric'in file_path:
        X = np.array(data.iloc[:, 1:])
    else:
        X = np.array(data.iloc[:, 1:-1])

    X = np.nan_to_num(stats.zscore(X))

    return X


def extract_project_edge(file_path):
    "Extract the edge in the project"

    if file_path.find('.csv') != -1:
        E_ = pd.read_csv(file_path, header=0, index_col=False)
    elif file_path.find('.emb') != -1:
        E_ = pd.read_csv(file_path, header=None, sep=" ", index_col=False)
    else:
        raise Exception('There is no way to handle files in formats other than csv and emb')

    E_ = np.array(E_.iloc[:, :]).tolist()

    E = []
    for i in E_:
        i = tuple(i)
        E.append(i)
    return E


def extra_project_label(file_path):
    "Extract the lable in the project"
    if file_path.find('.csv') != -1:
        data = pd.read_csv(file_path, header=0, index_col=False)
    else:
        raise Exception('There is no way to handle files in formats other than csv and emb')

    y = np.array(data['bug'])
    return y


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def concat_X(X_origin=None, X_vector=None, X_metric=None):
    """
    Construct a feature matrix, the vertical is the number of samples,
    the horizontal is the feature dimension of the sample, and the horizontal combination is performed
    """
    X = None
    if X_origin is not None:
        X = feature_concat(X, X_origin)
    if X_vector is not None:
        X = feature_concat(X, X_vector)
    if X_metric is not None:
        X = feature_concat(X, X_metric)

    if X is None:
        raise Exception(f'None feature used for model!')

    return X


def predata_and_G(project, mode, k):
    "Perform data preparation and hypergraph construction"
    X_origin = None
    X_vector = None
    X_metric = None
    if mode.find('origin') != -1:
        X_origin = extract_project_feature( 'data/'+project + "/Process-Binary.csv")
        X_origin = torch.Tensor(X_origin)
    if mode.find('vector') != -1:
        X_vector = extract_project_feature('data/'+project + "/Process-Vector.csv")
        X_vector = torch.Tensor(X_vector)
    if mode.find('metric') != -1:
        X_metric = extract_project_feature('data/'+ project + "/Process-Metric.csv")
        X_metric = torch.Tensor(X_metric)


    X = concat_X(X_origin, X_vector, X_metric)
    X = torch.Tensor(X)
    Y = extra_project_label('data/'+project + "/Process-Binary.csv")

    #construct code class dependency graph g
    E = extract_project_edge(  'data/'+project + "/dependencies_edges.csv")
    g = Graph(X.shape[0], E)
    # construct hypergragh G
    G = Hypergraph.from_graph(g)
    if mode.find('origin') != -1:
        G.add_hyperedges_from_feature_kNN(feature=X_origin, k=k)

    if mode.find('metric') != -1:
        G.add_hyperedges_from_feature_kNN(feature=X_metric, k=k)
    if mode.find('vector') != -1:
        G.add_hyperedges_from_feature_kNN(feature=X_vector, k=k)

    #return feature matrix:X , lable :Y , hypergraph :G
    return X, Y, G

