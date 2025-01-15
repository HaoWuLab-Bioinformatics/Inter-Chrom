from torch.utils.data import Dataset, random_split
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from imblearn.under_sampling import RandomUnderSampler


def get_name(cell_line, random_indices):
    name = []
    f = open('../data/' + cell_line + '/x.bed')
    for i in f.readlines():
        if i[0] != ' ':
            name.append(i.strip().split('\t')[0])
    f.close()
    name = name[random_indices]

    return name


def get_balance_dataset(x, y, genomics, chr_name, label):
    k = 10
    test_balance_dataset_k = []
    train_balance_dataset_k = []
    ratio_k = []
    gkf = GroupKFold(n_splits=k)
    for index_train, index_test in gkf.split(label, groups=chr_name):
        print(len(index_train))
        print(len(index_test))

        # Training set
        x_train = x[index_train]
        y_train = y[index_train]
        genomics_train = genomics[index_train]
        label_train = label[index_train]

        # Divide into positive and negative
        num_pos = np.sum(label_train == 1)
        num_neg = np.sum(label_train == 0)
        print(num_pos, num_neg)
        ratio = int(num_neg / num_pos)
        index_pos = np.where(label_train == 1)
        index_neg = np.where(label_train == 0)
        print('ratio:', ratio)

        # Positive data
        x_train_pos = x_train[index_pos]
        y_train_pos = y_train[index_pos]
        genomics_train_pos = genomics_train[index_pos]
        label_train_pos = label_train[index_pos]

        # Negative data
        x_train_neg_total = x_train[index_neg]
        y_train_neg_total = y_train[index_neg]
        genomics_train_neg_total = genomics_train[index_neg]
        label_train_neg_total = label_train[index_neg]

        # Test set
        x_test = x[index_test]
        y_test = y[index_test]
        genomics_test = genomics[index_test]
        label_test = label[index_test]

        # Balanced test set
        num = np.arange(0, len(label_test)).reshape(-1, 1)

        ros = RandomUnderSampler()
        num, label_test_bal = ros.fit_resample(num, label_test)
        num = np.squeeze(num).tolist()
        x_test_bal = x_test[num]
        y_test_bal = y_test[num]
        genomics_test_bal = genomics_test[num]

        kf = KFold(n_splits=ratio, shuffle=True)

        balance_dataset = []
        test_balance = []
        for _, index in kf.split(label_train_neg_total):
            x_train_neg = x_train_neg_total[index]
            y_train_neg = y_train_neg_total[index]
            genomics_train_neg = genomics_train_neg_total[index]
            label_train_neg = label_train_neg_total[index]

            x_train_kf = np.concatenate((x_train_pos, x_train_neg), axis=0)
            y_train_kf = np.concatenate((y_train_pos, y_train_neg), axis=0)
            genomics_train_kf = np.concatenate((genomics_train_pos, genomics_train_neg), axis=0)
            label_train_kf = np.concatenate((label_train_pos, label_train_neg))

            # Divide training set and validation set
            x_train_kf, x_val_kf, y_train_kf, y_val_kf, genomics_train_kf, genomics_val_kf, label_train_kf, label_val_kf = \
                train_test_split(
                            x_train_kf,
                            y_train_kf,
                            genomics_train_kf,
                            label_train_kf,
                            test_size=0.1,
                            random_state=0)

            balance_dataset.append((x_train_kf,
                                    y_train_kf,
                                    genomics_train_kf,
                                    label_train_kf,
                                    x_val_kf,
                                    y_val_kf,
                                    genomics_val_kf,
                                    label_val_kf))

        test_balance.append(x_test_bal)
        test_balance.append(y_test_bal)
        test_balance.append(genomics_test_bal)
        test_balance.append(label_test_bal)

        test_balance_dataset_k.append(test_balance)
        train_balance_dataset_k.append(balance_dataset)
        ratio_k.append(ratio)

    return train_balance_dataset_k, test_balance_dataset_k, ratio_k


def get_train_test_dataset(celltype):
    xf_feature1 = np.load('../token/' + celltype + '/xz_tokensid500' + '.npy')
    yf_feature1 = np.load('../token/' + celltype + '/yz_tokensid500' + '.npy')
    xf_feature2 = np.load('../token/' + celltype + '/xz_tokensid500_freq' + '.npy')
    yf_feature2 = np.load('../token/' + celltype + '/yz_tokensid500_freq' + '.npy')
    xr_feature1 = np.load('../token/' + celltype + '/xzr_tokensid500' + '.npy')
    yr_feature1 = np.load('../token/' + celltype + '/yzr_tokensid500' + '.npy')
    xr_feature2 = np.load('../token/' + celltype + '/xzr_tokensid500_freq' + '.npy')
    yr_feature2 = np.load('../token/' + celltype + '/yzr_tokensid500_freq' + '.npy')
    x_feature = np.stack((xf_feature1, xf_feature2, xr_feature1, xr_feature2), axis=1)
    y_feature = np.stack((yf_feature1, yf_feature2, yr_feature1, yr_feature2), axis=1)
    genomics = np.loadtxt('../data/' + celltype + '/genomics.csv', delimiter=',').astype(np.float32)
    label = np.loadtxt('../data/' + celltype + '/label.txt').astype(np.float32)
    name = []
    f = open('../data/' + celltype + '/x.bed')
    for i in f.readlines():
        if i[0] != ' ':
            name.append(i.strip().split('\t')[0])
    f.close()

    train_balance_dataset, test_balance_dataset, ratio = get_balance_dataset(x_feature, y_feature,
                                                                             genomics, name, label)

    return train_balance_dataset, test_balance_dataset, ratio


class GenomeDataset(Dataset):
    def __init__(self, dataset):
        self.seq_x, self.seq_y, self.genomics, self.label = dataset
        # print("init dataset")

    def __getitem__(self, idx):

        return self.seq_x[idx], self.seq_y[idx], self.genomics[idx], self.label[idx]

    def __len__(self):
        return len(self.seq_x)


