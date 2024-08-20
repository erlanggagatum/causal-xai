import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
import networkx as nx

from sklearn.model_selection import *
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from collections import defaultdict
import pickle as pkl

plt.rcParams["font.family"] = "Times New Roman"
import statistics

import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

import math
import bnlearn as bn

cuda_device = 'cuda:0'

# uncomment if you are using this library
# import torch
# import torch.autograd as autograd
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.init as init

# import tensorflow as tf
# import keras
# from keras import layers
# from keras import Model
# from keras.layers import *
# from keras.callbacks import EarlyStopping
# import keras.backend as K
# from keras.optimizers import Adam


# import shap
# import lime
# from lime import lime_tabular
# from tqdm import tqdm



# PREPROCESSING CODE

# === BASIC PREPROCESSING
def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, offset = 1):
    if target_column not in input_data.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the input data.")

    sequences = []
    data_size = len(input_data)

    for i in range(data_size - sequence_length - offset + 1):
        sequence = input_data.iloc[i:i + sequence_length]

        label_position = i + sequence_length
        label = input_data.iloc[label_position:label_position + offset][target_column]


        sequences.append((sequence, label))
    
    return sequences


def pd_import(file, filetype='csv'):
    return pd.read_csv(file, 
                          low_memory=False,
                          delimiter=',',
                          parse_dates=[0],
                          dayfirst=True,
                          index_col='datetime')
    
def scaler(dataset, mode='minmax'):
    
    dataset_values = np.copy(dataset.values.astype('float'))
    dataset_norm = dataset_values
    for c in range(8):
        dataset_norm[:, c] -= dataset_values[:, c].min()
        dataset_norm[:, c] /= (dataset_values[:, c].max()-dataset_values[:, c].min())
    dataset_norm = pd.DataFrame(dataset_norm, columns=dataset.columns, index=dataset.index)
    
    # dataset_norm = np.copy(dataset)
    # if mode == 'minmax': # default min max scaller
    #     num_cols = dataset.shape[1]
    #     for c in range(num_cols):
    #         dataset_norm[:, c] -= dataset[:, c].min()
    #         dataset_norm[:, c] /= (dataset[:, c].max()-dataset[:, c].min())
    #         if (max == 1 and min==0):
    #             print(c, 'passed')

    return dataset_norm


# === PEAK IDENTIFICATION PREPROCESSING
def callculate_Z_sn(sample_sequence, z_threshold = 0.5):
    N = len(sample_sequence)
    data = sample_sequence
    medians = []
    c = 1.1926
    alpha = 1.4285

    # calculate Sn
    for n in range(N):
        abs_diffs = [np.abs(data[n]) for m in range(N) if m != n]
        medians.append(np.median(abs_diffs))
    med_of_meds = np.median(medians)
    Sn = alpha * med_of_meds

    # Calculate all ZSn for all data points
    Z_sns = []
    for i in range(N):
        Z_Sn = np.abs(data[i] - np.median(data))
        Sn = alpha * med_of_meds
        Z_Sn = Z_Sn / Sn
        Z_sns.append(Z_Sn[0])
        
    Z_sns = np.array(Z_sns)
    Z_sns = (Z_sns > z_threshold).astype(int)
    
    return Z_sns

def Z_sn_preprocessing(sample_sequence):
    sequence = sample_sequence[0]['Global_active_power'].values
    sample_sequence = np.append(sequence, sample_sequence[1].values)
    sample_sequence = sample_sequence.reshape(len(sample_sequence), 1)
    return sample_sequence

def generate_peak_label(dataset_sequence, z_threshold = 0.5):
    peak_labels = []
    for data_sample in dataset_sequence:
        
        sample_sequence = Z_sn_preprocessing(data_sample)
        y_pred = callculate_Z_sn(sample_sequence, z_threshold=z_threshold)

        # print(Z_sns)
        if y_pred[-1] == 1:
            if sample_sequence[-1] > np.mean(sample_sequence):
                # Peak
                peak_labels.append(1)
                
            else:
                # Lower
                peak_labels.append(-1)
        else:
            # Normal
            peak_labels.append(0)
            
    print('finished generating peak labels..')
    if len(peak_labels) == len(dataset_sequence):
        print('peak_labels == dataset_sequence')
    else:
        raise Exception('Number of generated labels is not equal to len of dataset')
    return peak_labels


# === Discretized observed window observation
def de_mining_normalize_sequence(sequence):
    # sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    # return sequence
    
    # prepare data
    if not isinstance(sequence, np.ndarray) and isinstance(sequence, list):
        data = sequence.numpy()
    data = sequence
    # De-minning
    min_value = min(data)
    max_value = max(data)
    if min_value == 0 and max_value == 0:
        return sequence.tolist()
    
    de_minned_data = [x - min_value for x in data]

    # Normalization
    total_de_minned = sum(de_minned_data)
    normalized_data = [x / total_de_minned for x in de_minned_data]
    temp = np.array(normalized_data)
    # print(temp)
    if np.isnan(temp).any():
        normalized_data = np.nan_to_num(temp).tolist()
        print(normalized_data)
        # print(normalized_data)
        # print(de_minned_data)
        # print(sequence)
    
    return normalized_data

# calculate cumulative
def cumulative_sequence(sequence):
    new_sequences = []
    for point in sequence:
        if len(new_sequences) == 0:
            new_sequences.append(point)
        else:
            new_sequences.append(point + new_sequences[-1])
    return new_sequences

# wrapping cumulative sequence + dimining
def generate_cumulative_demined_sequence(dataset_sequence):
    norm_sequence = {}
    for sequence in dataset_sequence:
        columns = sequence[0].columns
        # print(sequence[0]['Sub_metering_1'].values)
        for col in columns:
            if col not in norm_sequence:
                norm_sequence[col] = []
            norm_sequence[col].append(de_mining_normalize_sequence(sequence[0][col].values))
        
    cumulative_seqs = {}
    for col in norm_sequence:
        if col not in cumulative_seqs:
            cumulative_seqs[col] = []
        for seq in norm_sequence[col]:
            if(max(seq) > 1.01):
                print(seq)
            cumulative_seqs[col].append(cumulative_sequence(seq))
            
    return norm_sequence, cumulative_seqs

# === Discretized x at window W
# Cluster cluster the cumulative sequence using kmeans
def cluster_cumulative_seqs(cumulative_seqs):
    col_cluster_labels = {}
    for col in cumulative_seqs:
        x = cumulative_seqs[col]
        print(len(x))
        wcss = []
        num_cluster = 3
        
        kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init="auto")
        cluster = kmeans.fit(X=x)
        labels = cluster.labels_
        centers = cluster.cluster_centers_
        
        # sort cluster based on its center
        centers_avg = centers.mean(axis=1)    
        print(centers_avg)
        sorted_kmeans_cluster_centers = np.sort(centers_avg)
        print(sorted_kmeans_cluster_centers)
        # ccenters[col] = sorted_kmeans_cluster_centers

        sorted_dic = {}
        for c in range(num_cluster):
            sorted_dic[c] = np.where(sorted_kmeans_cluster_centers == centers_avg[c])[0][0]
        print(sorted_dic)
        sorted_cluster = np.array([sorted_dic[c] for c in kmeans.labels_])
        # sorted_cluster
        col_cluster_labels[col] = [x, sorted_cluster, centers]
        
    return col_cluster_labels # [x, sorted_cluster, centers]

def generate_dataset_sequence_cluster(dataset_sequence, col_cluster_labels):
    
    dataset_sequence_cluster = []
    for i in range(len(dataset_sequence)):
        columns = dataset_sequence[i][0].columns
        
        data_temp = {}
        for col in columns:
            data_temp[col] = col_cluster_labels[col][1][i]
        data_temp = pd.DataFrame(data_temp, index=['cluster'])
        dataset_sequence_cluster.append(dataset_sequence[i] + (data_temp,))
        
    return dataset_sequence_cluster
    # dataset_sequence_cluster = generate_dataset_sequence_cluster(dataset_sequence_norm, col_cluster_labels)