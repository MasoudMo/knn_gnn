# Binary Classification of ECG Signals using Graph Neural Networks

## Summary
In this experiment, 15-node K-Nearest Neighbour (KNN) graphs are created with each node corresponding to one channel of ECG data. Graph Neural Networks (GNNs) are used to generate embeddings from these graphs, which are then used for classification. The purpose of this experiment is to explore alternatives to using medical metadata (such as patients' age or gender) in creating graphs prior to medical diagnosis tasks.

The preprocessing used to create the KNN graphs is illustrated in figure below:

![alt text](https://github.com/MasoudMo/knn_gnn/blob/master/docs/knn_gnn.PNG?raw=true)

## Usage
The dataset can directly be downloaded as a zip file from the following link: 
[PTB Diagnostic Databse](https://www.physionet.org/content/ptbdb/1.0.0/)

Alternatively, the download_data.py script can be used to download the dataset (Please note that the option above is a lot faster.):
```sh
python download_data.py --download_path <path to download dataset to. preferably "../data"> 
```

To train the model, the following command can be used (Please note that some options are left out purposefully. To see a complete list of available arguments, refer to src/train.py):
```sh
python train.py --best_model_path <path to save the trained model to> --data_path <path to the dataset> --k <number of neighbours in KNN> --epochs <number of training epochs>
```

To test the model, the following command can be used (Please note that some options are left out purposefully. To see a complete list of available arguments, refer to src/test.py):
```sh
python test.py --best_model_path <path to load the trained model from> --data_path <path to the dataset> --k <number of neighbours in KNN>
```