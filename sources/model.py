from dgl.nn.pytorch.conv import GATConv
import torch.nn.functional as F
import torch.nn as nn
import dgl


class GraphAttConvBinaryClassifier(nn.Module):
    """
    Classification model for the prostate cancer dataset using GAT
    """
    def __init__(self, in_dim, hidden_dim, num_classes, feat_drop=0, attn_drop=0, use_cuda=False):
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
        super(GraphAttConvBinaryClassifier, self).__init__()

        # Model layers
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop)
        self.conv2 = GATConv(hidden_dim, hidden_dim, num_heads=1, feat_drop=feat_drop, attn_drop=attn_drop)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, num_classes)

        self.use_cuda = use_cuda

    def forward(self, g):
        """
        Forward path of classifier
        Parameter:
            g (DGL Graph): Input graph
        """
        # Use RF signals as node features
        h = g.ndata['x']

        if self.use_cuda:
            h = h.cuda()

        # Two layers of Graph Convolution
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # Use the mean of hidden embeddings to find graph embedding
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')[0]

        # Fully connected output layer
        h = F.relu(self.fc_1(hg))
        out = self.fc_2(h)

        return out
