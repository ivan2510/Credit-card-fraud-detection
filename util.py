import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score

def preprocess_data(creditcard_filepath):
    df = pd.read_csv(creditcard_filepath)

    #Remove time because it does not carry any useful information
    df.drop('Time', axis = 1, inplace = True)

    #Scaling Amount values because it are too dispersed and it affects learning process
    sc = StandardScaler()
    amount = df['Amount'].values
    df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

    labels = df['Class'].values
    features = df.drop("Class", axis=1).values

    return features, labels

def dataset_split(dataset, batch_size):
    train_idx, test_idx= train_test_split(np.arange(len(dataset.y)), test_size=0.2, shuffle=True, stratify=dataset.y)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True)

    return train_loader, test_loader

def add_syntetic_data(features, labels):
    ada = ADASYN(sampling_strategy='minority', random_state=420, n_neighbors = 7)
    features_new, labels_new = ada.fit_resample(features, labels)

    return features_new, labels_new

def draw_line_plot(epoch_list, values_train, values_test, title):
    plt.plot(epoch_list, values_train, label = "Train")
    plt.plot(epoch_list, values_test, label = "Test")
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(os.getcwd(), title + ".png"))

def draw_confusion_matrix(predictions, targets):
    conf_matrix = confusion_matrix(y_true=targets, y_pred=predictions)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Targets', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(os.getcwd(), 'confusion_matrix.png'))


if __name__ == '__main__':
    a = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    b = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1]

    draw_confusion_matrix(b, a)
