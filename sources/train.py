from ecg_gnn.sources.data import PtbEcgDataSet
from torch.utils.data import DataLoader
from ecg_gnn.sources.model import GraphAttConvBinaryClassifier
import logging
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from ecg_gnn.sources.data import  collate
import numpy as np


logger_level = logging.INFO

logger = logging.getLogger('gnn_prostate_cancer')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)


def main():
    dataset = PtbEcgDataSet(
        root_dir='H:\Workspace\ELEC 421 Project\data\ptb-diagnostic-ecg-database-1.0.0\ptb-diagnostic-ecg-database-1.0.0',
        records='RECORDS',
        weighted=False)

    dataset_train_len = len(dataset)
    logger.info("Training dataset has {} samples".format(dataset_train_len))

    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate)

    model = GraphAttConvBinaryClassifier(in_dim=31000,
                                         hidden_dim=300,
                                         num_classes=15,
                                         use_cuda=False,
                                         feat_drop=0,
                                         attn_drop=0)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = list()
    train_accs = list()
    loss = 0

    for epoch in range(100):

        y_true = list()
        y_pred = list()

        epoch_loss = 0
        for bg, label in dataloader:

            y_true.append(label.numpy()[0])

            prediction = model(bg)

            y_pred.append(np.argmax(prediction.detach().numpy(), axis=1)[0])

            loss = loss_func(prediction, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.detach().item()

        epoch_loss /= dataset_train_len
        logger.info('Training epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        train_losses.append(epoch_loss)
        acc = accuracy_score(y_true, y_pred)
        logger.info("Training accuracy {:.4f}".format(acc))
        train_accs.append(acc)


if __name__ == "__main__":
    main()