from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.conv import SGConv
from dgl.nn.pytorch.conv import ChebConv
import torch.nn.functional as F
import torch.nn as nn
import dgl
import torch


class GraphConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, fc_hidden_1, fc_hidden_2, num_classes, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            num_classes (int): Number of output classes
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GraphConv(in_dim, hidden_dim_1)
        self.conv2 = GraphConv(hidden_dim_1, hidden_dim_2)
        self.conv3 = GraphConv(hidden_dim_2, fc_hidden_1)

        self.fc_1 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.fc_2 = nn.Linear(fc_hidden_2, num_classes)
        self.out = nn.Sigmoid()

        self.use_cuda = use_cuda

    def forward(self, g, h):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))

        with g.local_scope():
            # Use the mean of hidden embeddings to find graph embedding
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')

            # Fully connected output layer
            h = F.relu(self.fc_1(hg))
            h = self.fc_2(h)
            out = self.out(h)

            return torch.squeeze(out, 1)


class SimpleGraphConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, fc_hidden_1, fc_hidden_2, num_classes, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            num_classes (int): Number of output classes
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(SimpleGraphConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = SGConv(in_dim, hidden_dim_1)
        self.conv2 = SGConv(hidden_dim_1, hidden_dim_2)
        self.conv3 = SGConv(hidden_dim_2, fc_hidden_1)

        self.fc_1 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.fc_2 = nn.Linear(fc_hidden_2, num_classes)

        self.out = nn.LogSoftmax(dim=1)

        self.use_cuda = use_cuda

    def forward(self, g, h):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))

        with g.local_scope():
            # Use the mean of hidden embeddings to find graph embedding
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')

            # Fully connected output layer
            h = F.relu(self.fc_1(hg))
            h = self.fc_2(h)
            out = self.out(h)

            return torch.squeeze(out, 1)


class ChebConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, num_classes, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            num_classes (int): Number of output classes
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(ChebConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = ChebConv(in_dim, hidden_dim_1, k=5)
        self.conv2 = ChebConv(hidden_dim_1, hidden_dim_2, k=5)
        self.conv3 = ChebConv(hidden_dim_2, hidden_dim_2, k=5)

        self.fc_1 = nn.Linear(hidden_dim_2, hidden_dim_2)
        self.fc_2 = nn.Linear(hidden_dim_2, num_classes)

        self.out = nn.LogSoftmax(dim=1)

        self.use_cuda = use_cuda

    def forward(self, g, h):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))

        with g.local_scope():
            # Use the mean of hidden embeddings to find graph embedding
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')

            # Fully connected output layer
            h = F.relu(self.fc_1(hg))
            h = self.fc_2(h)
            out = self.out(h)

            return torch.squeeze(out, 1)


class GraphAttentionConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim_1, num_classes, feat_drop=0, attn_drop=0, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            num_classes (int): Number of output classes
            feat_drop (float): Indicates the dropout rate for features
            attn_drop (float): Indicates the dropout rate for the attention mechanism
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphAttentionConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GATConv(in_dim, hidden_dim_1, feat_drop=feat_drop, attn_drop=attn_drop, num_heads=1)
        self.conv2 = GATConv(hidden_dim_1, hidden_dim_1, feat_drop=feat_drop, attn_drop=attn_drop, num_heads=1)

        self.fc_1 = nn.Linear(hidden_dim_1, num_classes)

        self.out = nn.LogSoftmax(dim=1)

        self.use_cuda = use_cuda

    def forward(self, g, h):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        with g.local_scope():
            # Use the mean of hidden embeddings to find graph embedding
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')

            # Fully connected output layer
            h = self.fc_1(hg)
            out = self.out(h)

            return torch.squeeze(out, 1)


class GraphSageBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim_1, fc_hidden_1, fc_hidden_2, hidden_dim_2, num_classes, feat_drop=0, use_cuda=False):
        """
        Constructor for the GraphAttConvBinaryClassifier class
        Parameters:
            in_dim (int): Dimension of features for each node
            hidden_dim (int): Dimension of hidden embeddings
            num_classes (int): Number of output classes
            use_cuda (bool): Indicates whether GPU should be utilized or not
        """
        super(GraphSageBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = SAGEConv(in_dim, hidden_dim_1, feat_drop=feat_drop, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_dim_1, hidden_dim_2, feat_drop=feat_drop, aggregator_type='mean')
        self.conv3 = SAGEConv(hidden_dim_2, fc_hidden_1, feat_drop=feat_drop, aggregator_type='mean')

        self.fc_1 = nn.Linear(fc_hidden_1, fc_hidden_2)
        self.fc_2 = nn.Linear(fc_hidden_2, num_classes)

        self.out = nn.LogSoftmax(dim=1)

        self.use_cuda = use_cuda

    def forward(self, g, h):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))

        with g.local_scope():
            # Use the mean of hidden embeddings to find graph embedding
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')

            # Fully connected output layer
            h = F.relu(self.fc_1(hg))
            h = self.fc_2(h)
            out = self.out(h)

            return torch.squeeze(out, 1)


