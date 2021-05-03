"""
Author: Masoud Mokhtari

Trains a Graph Neural Network (GNN) model on K-Nearest Neighbour (KNN) graphs created using Fast Fourier Transform (FFT)
of ECG signals. Each graph contains 15 nodes with each node corresponding to one channel of ECG data.

"""

from torch.utils.data import DataLoader
from model import GraphConvBinaryClassifier
from model import GraphSageBinaryClassifier
from model import SimpleGraphConvBinaryClassifier
from model import GraphAttentionConvBinaryClassifier
from data import collate
from data import PtbEcgDataSet
import logging
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import random_split
import visdom
from sklearn.metrics import roc_auc_score
import argparse


logger_level = logging.INFO

logger = logging.getLogger('ecg_gnn')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)


def main():

    # Command line argument parser
    parser = argparse.ArgumentParser(description='GNN training on ECG dataset')
    parser.add_argument('--best_model_path',
                        type=str,
                        required=False,
                        default="../models/model.pt",
                        help='Path to save the best model to (Should contain the file name. e.g model.py)')
    parser.add_argument('--history_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Path to save training history to.')
    parser.add_argument('--visualize',
                        type=bool,
                        required=False,
                        default=None,
                        help='Indicates whether visdom plots are created or not.')
    parser.add_argument('--data_path',
                        type=str,
                        required=False,
                        default='../data/',
                        help='Path to dataset')
    parser.add_argument('--k',
                        type=int,
                        required=False,
                        default=3,
                        help='Number of neighbours to use for KNN graph')
    parser.add_argument('--samp_from',
                        type=int,
                        required=False,
                        default=0,
                        help='Data sample index to start from.')
    parser.add_argument('--samp_to',
                        type=int,
                        required=False,
                        default=5000,
                        help='Data sample index to end at.')
    parser.add_argument('--fft_graph',
                        type=bool,
                        required=False,
                        default=None,
                        help='Indicates whether fft data is used for graph creation')
    parser.add_argument('--filter',
                        type=bool,
                        required=False,
                        default=None,
                        help='Indicates whether data is filtered')
    parser.add_argument('--fft_data',
                        type=bool,
                        required=False,
                        default=None,
                        help='Indicates whether fft data is used for node features')
    parser.add_argument('--pca_dim',
                        type=int,
                        required=False,
                        default=15,
                        help='Indicates the dimension to use for PCA.')
    parser.add_argument('--test_split',
                        type=float,
                        required=False,
                        default=0.1,
                        help='Indicates the test split percentage.')
    parser.add_argument('--val_split',
                        type=float,
                        required=False,
                        default=0.1,
                        help='Indicates the validation split percentage.')
    parser.add_argument('--batch_size',
                        type=int,
                        required=False,
                        default=10,
                        help='Indicates the number of graphs per batch.')
    parser.add_argument('--learning_rate',
                        type=float,
                        required=False,
                        default=0.01,
                        help='Indicates learning rate.')
    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=100,
                        help='Indicates the number of epochs')
    parser.add_argument('--hidden_dim_1',
                        type=int,
                        required=False,
                        default=15,
                        help='Indicates the dimension of first conv hidden layer')
    parser.add_argument('--hidden_dim_2',
                        type=int,
                        required=False,
                        default=10,
                        help='Indicates the dimension of second conv hidden layer')
    parser.add_argument('--fc_hidden_1',
                        type=int,
                        required=False,
                        default=10,
                        help='Indicates the dimension of first FC hidden layer')
    parser.add_argument('--fc_hidden_2',
                        type=int,
                        required=False,
                        default=10,
                        help='Indicates the dimension of second FC hidden layer')
    parser.add_argument('--input_dim',
                        type=int,
                        required=False,
                        default=15,
                        help='Indicates the dimension of input layer')
    parser.add_argument('--fft_points_to_keep',
                        type=int,
                        required=False,
                        default=500,
                        help='number of fft samples to keep')
    args = parser.parse_args()

    history_path = args.history_path
    best_model_path = args.best_model_path
    visualize = args.visualize
    data_path = args.data_path
    k = args.k
    samp_from = args.samp_from
    samp_to = args.samp_to
    fft_graph = args.fft_graph
    fft_data = args.fft_data
    pca_dim = args.pca_dim
    test_split = args.test_split
    val_split = args.val_split
    batch_size = args.batch_size
    lr = args.learning_rate
    hidden_dim_1 = args.hidden_dim_1
    hidden_dim_2 = args.hidden_dim_2
    input_dim = args.input_dim
    fft_points_to_keep = args.fft_points_to_keep
    fc_hidden_1 = args.fc_hidden_1
    fc_hidden_2 = args.fc_hidden_2
    epochs = args.epochs
    use_filter = args.filter

    # Set torch seed for reproducability
    torch.manual_seed(10)

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    if visualize:
        vis = visdom.Visdom()

    # Load dataset
    dataset = PtbEcgDataSet(
        root_dir=data_path,
        records='RECORDS',
        weighted=False,
        k=k,
        n_jobs=5,
        fft_data=fft_data,
        filter=use_filter,
        fft_graph=fft_graph,
        samp_from=samp_from,
        samp_to=samp_to,
        fft_points_to_keep=fft_points_to_keep,
        pca_dim=pca_dim)

    # Split data into test, validation and training sets
    dataset_len = len(dataset)
    val_dataset_len = int(val_split*(1-test_split)*dataset_len)
    test_dataset_len = int(test_split*dataset_len)
    train_dataset_len = dataset_len - val_dataset_len - test_dataset_len

    if test_split != 0:
        train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                                [train_dataset_len, val_dataset_len, test_dataset_len])
        # Create data loaders
        train_dataloader = DataLoader(train_dataset,
                                      shuffle=True, collate_fn=collate, batch_size=batch_size, drop_last=True)
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=True, collate_fn=collate, batch_size=batch_size, drop_last=True)
        test_dataloader = DataLoader(test_dataset,
                                     shuffle=True, collate_fn=collate, batch_size=batch_size, drop_last=True)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_dataset_len, val_dataset_len])
        # Create data loaders
        train_dataloader = DataLoader(train_dataset,
                                      shuffle=True, collate_fn=collate, batch_size=batch_size, drop_last=True)
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=True, collate_fn=collate, batch_size=batch_size, drop_last=True)

    logger.info("Training dataset has {} samples".format(train_dataset_len))
    logger.info("Validation dataset has {} samples".format(val_dataset_len))
    logger.info("Test dataset has {} samples".format(test_dataset_len))

    # Create the model
    model = GraphConvBinaryClassifier(in_dim=input_dim,
                                      hidden_dim_1=hidden_dim_1,
                                      hidden_dim_2=hidden_dim_2,
                                      num_classes=1,
                                      use_cuda=False,
                                      fc_hidden_1=fc_hidden_1,
                                      fc_hidden_2=fc_hidden_2)

    # Define the loss function
    loss_func = nn.BCELoss()

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = list()
    train_accs = list()

    val_losses = list()
    val_accs = list()

    test_losses = list()
    test_accs = list()

    max_val_acc = 0

    for epoch in range(epochs):

        # Enter training mode
        model.train()

        y_true = list()
        y_pred = list()
        epoch_loss = 0

        for bg, label in train_dataloader:

            # Move label and graph to GPU if available
            if use_cuda:
                torch.cuda.empty_cache()
                label = label.to(device)

            # Gathering true labels
            y_true.append(label.numpy().flatten())

            # Obtain node features for the batch
            features = bg.ndata['x'].float()

            # Get model prediction
            prediction = model(bg, features)

            # Gather model predictions
            y_pred.append(prediction.detach().numpy().flatten())

            # Compute loss
            loss = loss_func(prediction, label)

            # Reset gradients
            optimizer.zero_grad()

            # Compute the gradients (Backprop)
            loss.backward()

            # Perform one optimization step
            optimizer.step()

            # Add to epoch loss
            epoch_loss += loss.detach().item()

        epoch_loss /= train_dataset_len
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        logger.info('Training epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        train_losses.append(epoch_loss)

        # Compute simple accuracy
        acc = roc_auc_score(y_true, y_pred)
        logger.info("Training epoch {}, accuracy {:.4f}".format(epoch, acc))
        train_accs.append(acc)

        # plot the loss
        if visualize:
            vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                     update='append', win='tain_loss',
                     opts=dict(title="Train Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

            vis.line(Y=torch.reshape(torch.tensor(acc), (-1, )), X=torch.reshape(torch.tensor(epoch), (-1, )),
                     update='append', win='train_acc',
                     opts=dict(title="Train Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

        # Save to history if needed
        if history_path:
            f = open(history_path+"_train_losses.txt", "w")
            for ele in train_losses:
                f.write(str(ele) + "\n")
            f.close()

            f = open(history_path + "_train_accs.txt", "w")
            for ele in train_accs:
                f.write(str(ele) + "\n")
            f.close()

        # Enter evaluation mode
        model.eval()

        with torch.no_grad():
            y_true = list()
            y_pred = list()
            epoch_loss = 0

            for bg, label in val_dataloader:

                # Move label and graph to GPU if available
                if use_cuda:
                    torch.cuda.empty_cache()
                    label = label.to(device)

                y_true.append(label.detach().numpy().flatten())

                # Obtain node features for the batch
                features = bg.ndata['x'].float()

                # Get model prediction
                prediction = model(bg, features)

                # Gather model predictions
                y_pred.append(prediction.detach().numpy().flatten())

                # Compute loss
                loss = loss_func(prediction, label)

                # Accumulate validation loss
                epoch_loss += loss.detach().item()

            epoch_loss /= val_dataset_len
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()

            logger.info('Validation epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            val_losses.append(epoch_loss)

            # Compute simple accuracy
            acc = roc_auc_score(y_true, y_pred)
            logger.info("Validation epoch {}, accuracy {:.4f}".format(epoch, acc))
            val_accs.append(acc)

            # plot the loss
            if visualize:
                vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='val_loss',
                         opts=dict(title="Validation Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                vis.line(Y=torch.reshape(torch.tensor(acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                         update='append', win='val_acc',
                         opts=dict(title="Validation Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

            # Save to history if needed
            if history_path:
                f = open(history_path + "_val_losses.txt", "w")
                for ele in val_losses:
                    f.write(str(ele) + "\n")
                f.close()

                f = open(history_path + "_val_accs.txt", "w")
                for ele in val_accs:
                    f.write(str(ele) + "\n")
                f.close()

            if acc > max_val_acc:
                max_val_acc = acc
                # Save model checkpoint if validation accuracy has increased
                logger.info("Validation accuracy increased. Saving model to {}".format(best_model_path))
                torch.save(model.state_dict(), best_model_path)

            if test_split != 0:
                y_true = list()
                y_pred = list()
                epoch_loss = 0

                for bg, label in test_dataloader:

                    # Move label and graph to GPU if available
                    if use_cuda:
                        torch.cuda.empty_cache()
                        label = label.to(device)

                    # Gather true labels
                    y_true.append(label.detach().numpy().flatten())

                    # Obtain node features for the batch
                    features = bg.ndata['x'].float()

                    # Get model prediction
                    prediction = model(bg, features)

                    # Gather model predictions
                    y_pred.append(prediction.detach().numpy().flatten())

                    # Compute loss
                    loss = loss_func(prediction, label)

                    # Accumulate validation loss
                    epoch_loss += loss.detach().item()

                epoch_loss /= test_dataset_len
                y_true = np.array(y_true).flatten()
                y_pred = np.array(y_pred).flatten()

                logger.info('Test epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
                test_losses.append(epoch_loss)

                # Compute simple accuracy
                acc = roc_auc_score(y_true, y_pred)
                logger.info("Test epoch {}, accuracy {:.4f}".format(epoch, acc))
                test_accs.append(acc)

                # plot the loss
                if visualize:
                    vis.line(Y=torch.reshape(torch.tensor(epoch_loss), (-1,)),
                             X=torch.reshape(torch.tensor(epoch), (-1,)),
                             update='append', win='test_loss',
                             opts=dict(title="Test Losses Per Epoch", xlabel="Epoch", ylabel="Loss"))

                    vis.line(Y=torch.reshape(torch.tensor(acc), (-1,)), X=torch.reshape(torch.tensor(epoch), (-1,)),
                             update='append', win='test_acc',
                             opts=dict(title="Test Accuracy Per Epoch", xlabel="Epoch", ylabel="Accuracy"))

                # Save to history if needed
                if history_path:
                    f = open(history_path + "_test_losses.txt", "w")
                    for ele in test_losses:
                        f.write(str(ele) + "\n")
                    f.close()

                    f = open(history_path + "_test_accs.txt", "w")
                    for ele in test_accs:
                        f.write(str(ele) + "\n")
                    f.close()


if __name__ == "__main__":
    main()
