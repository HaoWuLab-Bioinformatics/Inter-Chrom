import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import shutil
import numpy as np
import get_dataset as get_dataset
from model import InterChrom
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, \
    accuracy_score, matthews_corrcoef

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


class ArgsNamespace:
    def __init__(self, dataloader_ddp_disabled):
        self.dataloader_ddp_disabled = dataloader_ddp_disabled
        self.run_seed = None
        self.run_save_path = 'checkpoints'
        self.dataset_data_root = 'data'
        self.dataset_celltype = 'GM12878'
        self.model_type = 'MVAE'
        self.trainer_patience = 10
        self.trainer_max_epochs = 200
        self.dataloader_batch_size = 32
        self.lr = 1e-4
        self.resume = ''
        self.result_dir = './' + self.dataset_celltype + '_model/VAEResult_' + str(self.trainer_max_epochs) + '_' + \
                          str(self.dataloader_batch_size)
        self.save_dir = './' + self.dataset_celltype + '_model/checkPoint_' + str(self.trainer_max_epochs) + '_' + \
                        str(self.dataloader_batch_size)


def init_parser():
    args = ArgsNamespace(dataloader_ddp_disabled=False)
    return args


def dataset(args):
    train_balance_dataset, test_balance_dataset, ratio = get_dataset.get_train_test_dataset(args.dataset_celltype)
    train_dataset_k = []
    val_dataset_k = []
    tmp = 0
    for i in range(len(train_balance_dataset)):
        train_dataset = list(range(ratio[i]))
        val_dataset = list(range(ratio[i]))
        for idx in range(len(train_balance_dataset[i])):
            seq_x_train, seq_y_train, genomics_train, label_train, seq_x_val, seq_x_val, genomics_val, label_val = \
            train_balance_dataset[i][idx]
            train_dataset[idx] = seq_x_train, seq_y_train, genomics_train, label_train
            val_dataset[idx] = seq_x_val, seq_x_val, genomics_val, label_val

            tmp = len(genomics_val[0])
        train_dataset_k.append(train_dataset)
        val_dataset_k.append(val_dataset)

    return train_dataset_k, val_dataset_k, test_balance_dataset, tmp


def save_checkpoint(state, is_best, outdir):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def test(args, model, optimizer, val_dataset_idx, i, epoch, best_test_loss, patience):
    with torch.no_grad():
        seq_x = torch.tensor(val_dataset_idx[i][0]).to(device)
        seq_y = torch.tensor(val_dataset_idx[i][1]).to(device)
        # genomics = torch.tensor(val_dataset_idx[i][2]).to(device)
        label = torch.tensor(val_dataset_idx[i][3]).to(device)
        outputs = model(seq_x, seq_y)
        outputs = outputs.squeeze(-1)
        criterion = torch.nn.BCELoss()
        loss = criterion(outputs, label)

        is_best = (loss) < best_test_loss
        if is_best:
            patience = 0
        else:
            patience += 1
        best_test_loss = min(loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir)

        return best_test_loss, patience


def main():
    args = init_parser()
    train_dataset, val_dataset, test_dataset, genomics_dim = dataset(args)
    num = len(train_dataset)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    auprc = np.zeros(num)
    acc = np.zeros(num)
    mcc = np.zeros(num)
    precision = np.zeros(num)
    recall = np.zeros(num)
    f1 = np.zeros(num)

    for idx in range(num):
        train_dataset_idx = train_dataset[idx]
        val_dataset_idx = val_dataset[idx]
        current_result_per_per = np.zeros((len(train_dataset_idx), len(test_dataset[idx][0])))
        test_seq_x = test_dataset[idx][0]
        test_seq_y = test_dataset[idx][1]
        # test_genomics = test_dataset[idx][2]
        test_label = test_dataset[idx][3]

        for i in range(len(train_dataset_idx)):
            start_epoch = 0
            model = InterChrom().to(device)
            train_dataloader = get_dataset.GenomeDataset(train_dataset_idx[i])
            train_dataloader = DataLoader(train_dataloader, batch_size=args.dataloader_batch_size, shuffle=True, drop_last=True)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            best_test_loss = np.finfo('f').max

            patience = 0
            for epoch in range(start_epoch, args.trainer_max_epochs):
                for batch_index, x in enumerate(train_dataloader):
                    seq_x = x[0].to(device)
                    seq_y = x[1].to(device)
                    # genomics = x[2].to(device)
                    label = x[3].to(device)
                    label_outputs = model(seq_x, seq_y)
                    criterion = torch.nn.BCELoss()
                    label_loss = criterion(label_outputs.squeeze(-1), label)
                    optimizer.zero_grad()
                    label_loss.backward()
                    optimizer.step()

                    # if (batch_index + 1) == len(train_dataloader):
                    #     print('[{}/{}], [{}/{}], Epoch [{}/{}], '
                    #           'Batch [{}/{}] : label-loss = {:.4f}, '
                    #           'best_loss = {:.4f}'
                    #           .format(idx + 1, num, i + 1, len(train_dataset_idx), epoch + 1,
                    #                   args.trainer_max_epochs, batch_index + 1, len(train_dataloader),
                    #                   label_loss.item(), best_test_loss))

                best_test_loss, patience = test(args, model, optimizer, val_dataset_idx, i, epoch,
                                                best_test_loss, patience)
                if (patience > args.trainer_patience) or ((epoch + 1) == args.trainer_max_epochs):
                    break

            if args.save_dir:
                checkpoint_path = args.save_dir + '/model_best.pth'
                if os.path.isfile(checkpoint_path):
                    # print('=> loading checkpoint %s' % checkpoint_path)
                    checkpoint = torch.load(checkpoint_path)
                    # start_epoch = checkpoint['epoch'] + 1
                    # best_test_loss = checkpoint['best_test_loss']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    # print('=> loaded checkpoint %s' % checkpoint_path)
                else:
                    print('=> no checkpoint found at %s' % checkpoint_path)
            model.eval()

            current_result_per_per[i] = model(torch.tensor(test_seq_x).to(device),
                                          torch.tensor(test_seq_y).to(device)).squeeze(-1).cpu().detach().numpy()

        current_result_per = current_result_per_per.mean(axis=0)
        current_result_per_class = np.around(current_result_per)

        # individual Evaluation
        auprc[idx], acc[idx], mcc[idx] = EvaluateMetrics(test_label, current_result_per_class, current_result_per)
        precision[idx], recall[idx], f1[idx], _ = precision_recall_fscore_support(test_label,
                                                                                  current_result_per_class,
                                                                                  average='binary')

    # auprc_mean = auprc.mean(axis=0)
    # acc_mean = acc.mean(axis=0)
    # mcc_mean = mcc.mean(axis=0)
    # precision_mean = precision.mean(axis=0)
    # recall_mean = recall.mean(axis=0)
    # f1_mean = f1.mean(axis=0)
    #
    # print('10-fold_cross-validation')
    # print('auprc_bal:', auprc_mean)
    # print('acc_bal:', acc_mean)
    # print('mcc_bal:', mcc_mean)
    # print('precision_bal:', precision_mean)
    # print('recall_bal', recall_mean)
    # print('f1_bal:', f1_mean)
    # print('----------------------------------------------------------------')
    # for i in range(num):
    #     print(auprc[i], acc[i], mcc[i], precision[i], recall[i], f1[i])

    return None


def EvaluateMetrics(y_test, label, proba):
    auprc = average_precision_score(y_test, proba)
    acc = accuracy_score(y_test, label)
    mcc = matthews_corrcoef(y_test, label)
    return auprc, acc, mcc


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
        main()
