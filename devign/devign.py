import torch
import torch.nn as nn
from torch_geometric.nn.conv import GatedGraphConv

class Devign(nn.Module):

    def __init__(self, gated_graph_conv_args, conv_args, emb_size: int):
        super(Devign, self).__init__()

        # Gate Graph Convolution Layer
        self.ggc = GatedGraphConv(**gated_graph_conv_args)
        
        #Convolutional layers
        self.conv1d_1 = nn.Conv1d(**conv_args["conv1d_1"])
        self.bn1 = nn.BatchNorm1d(conv_args["conv1d_1"]['out_channels'])
        self.conv1d_2 = nn.Conv1d(**conv_args["conv1d_2"])
        self.bn2 = nn.BatchNorm1d(conv_args["conv1d_2"]['out_channels'])

        # Activation layers
        self.relu1 = nn.LeakyReLU()

        # Dense Linear layers
        fc1_size = gated_graph_conv_args["out_channels"] + emb_size
        fc1_size = self.get_conv_mp_out_size(fc1_size, conv_args["conv1d_2"], [conv_args["maxpool1d_1"], conv_args["maxpool1d_2"]])
        fc2_size = gated_graph_conv_args["out_channels"]
        fc2_size = self.get_conv_mp_out_size(fc2_size, conv_args["conv1d_2"], [conv_args["maxpool1d_2"], conv_args["maxpool1d_2"]])
        self.fc1 = nn.Linear(in_features=fc1_size, out_features=1)
        self.fc2 = nn.Linear(in_features=fc2_size, out_features=1)

        # Dropout
        # self.drop = nn.Dropout(p=0.2)

        # Max pooling layers
        self.mp_1 = nn.MaxPool1d(**conv_args["maxpool1d_1"])
        self.mp_2 = nn.MaxPool1d(**conv_args["maxpool1d_2"])

        # Number of model's parameters
        self.count_parameters()

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # GGC
        hidden = self.ggc(x, edge_index)
        # Convolutional modules
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1.in_channels, concat_size)

        Z = self.mp_1(self.relu1(self.bn1(self.conv1d_1(concat))))
        Z = self.mp_2(self.bn2(self.conv1d_2(Z)))

        hidden = hidden.view(-1, self.conv1d_1.in_channels, hidden.shape[1])
        Y = self.mp_1(self.relu1(self.bn1(self.conv1d_1(hidden))))
        Y = self.mp_2(self.bn2(self.conv1d_2(Y)))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        
        # res = self.drop(res)
        
        sig = torch.sigmoid(torch.flatten(res))

        return sig

    def save(self, path=None) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        self.load_state_dict(torch.load(path, weights_only=True))

    def count_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")

    def get_conv_mp_out_size(self, in_size, last_layer, mps):
        size = in_size

        for mp in mps:
            size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

        size = size + 1 if size % 2 != 0 else size

        return int(size * last_layer["out_channels"])
    
    def get_embedding(self, data):
        x, edge_index = data.x, data.edge_index

        hidden = self.ggc(x, edge_index)
        concat = torch.cat([hidden, x], 1)
        concat_size = concat.shape[1]
        concat = concat.view(-1, self.conv1d_1.in_channels, concat_size)

        Z = self.mp_1(self.relu1(self.bn1(self.conv1d_1(concat))))
        Z = self.mp_2(self.bn2(self.conv1d_2(Z)))

        hidden = hidden.view(-1, self.conv1d_1.in_channels, hidden.shape[1])
        Y = self.mp_1(self.relu1(self.bn1(self.conv1d_1(hidden))))
        Y = self.mp_2(self.bn2(self.conv1d_2(Y)))

        Z = Z.view(Z.size(0), -1)
        Y = Y.view(Y.size(0), -1)

        embedding = torch.cat([Z, Y], dim=1)
        
        return embedding  # shape: [1, Z+Y dim]
