
from torch.utils.data import DataLoader
from model import GraphConvBinaryClassifier
from model import GraphSageBinaryClassifier
from model import SimpleGraphConvBinaryClassifier
from model import GraphAttentionConvBinaryClassifier
import logging
import torch.nn as nn
from data import collate
from data import PtbEcgDataSet
import numpy as np
import torch
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
                        default="./model.pt",
                        help='Path to save the best model to (Should contain the file name. e.g model.py)')
    parser.add_argument('--data_path',
                        type=str,
                        required=False,
                        default='H:\Workspace\ELEC 421\ecg_gnn\data\ptb-diagnostic-ecg-database-1.0.0\ptb-diagnostic-ecg-database-1.0.0',
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
                        default=None,
                        help='Indicates the dimension to use for PCA.')
    parser.add_argument('--fft_points_to_keep',
                        type=int,
                        required=False,
                        default=500,
                        help='number of fft samples to keep')
    args = parser.parse_args()

    best_model_path = args.best_model_path
    data_path = args.data_path
    k = args.k
    samp_from = args.samp_from
    samp_to = args.samp_to
    fft_graph = args.fft_graph
    fft_data = args.fft_data
    pca_dim = args.pca_dim
    fft_points_to_keep = args.fft_points_to_keep
    use_filter = args.filter

    # Set torch seed for reproducability
    torch.manual_seed(10)

    # Check whether cuda is available or not
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

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

    test_dataloader = DataLoader(dataset,
                                 shuffle=True, collate_fn=collate)

    logger.info("Test dataset has {} samples".format(dataset_len))

    # Create the model
    model = GraphConvBinaryClassifier(in_dim=15,
                                      hidden_dim_1=12,
                                      hidden_dim_2=6,
                                      num_classes=1,
                                      use_cuda=False,
                                      fc_hidden_1=3,
                                      fc_hidden_2=2)

    # Define the loss function
    loss_func = nn.BCELoss()

    # Enter evaluation mode
    model.eval()

    with torch.no_grad():
        y_true = list()
        y_pred = list()
        epoch_loss = 0

        for bg, label in test_dataloader:

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

        epoch_loss /= dataset_len
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        logger.info('Total Loss {:.4f}'.format(epoch_loss))
        acc = roc_auc_score(y_true, y_pred)
        logger.info("Total accuracy {:.4f}".format(acc))

        for i, pred in enumerate(y_pred):
            logger.info("True label is: {} and prediction is: {}".format(y_true[i], pred))


if __name__ == "__main__":
    main()