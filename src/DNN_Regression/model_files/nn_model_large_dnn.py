import torch.nn as nn

############## Define Deep Regression Model ##################
class DeepReg(nn.Module):
    def __init__(self, layer_num, input_dim, hidden_dim, output_dim):
        super(DeepReg, self).__init__()
        
        self.deepreg_layers = nn.ModuleList()
        cur_layer = nn.Linear(in_features=input_dim,out_features=64,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(64))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=64,out_features=32,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(32))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=32,out_features=16,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(16))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=16,out_features=8,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(8))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=8,out_features=1,bias=True)
        self.deepreg_layers.append(cur_layer)
        # for i in range(layer_num):
        #     if i==0: #--- the input layer
        #         cur_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        #         self.deepreg_layers.append(cur_layer)
        #         self.deepreg_layers.append(nn.BatchNorm1d(hidden_dim))
        #         self.deepreg_layers.append(nn.LeakyReLU())
        #         self.deepreg_layers.append(nn.Dropout(p=0.5))
        #     elif i== layer_num -1: #----the last layer
        #         cur_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        #         self.deepreg_layers.append(cur_layer)
        #     else: #---- the other hidden layers
        #         cur_layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        #         self.deepreg_layers.append(cur_layer)
        #         self.deepreg_layers.append(nn.BatchNorm1d(hidden_dim))
        #         self.deepreg_layers.append(nn.LeakyReLU())
        #         self.deepreg_layers.append(nn.Dropout(p=0.5))
                

    def forward(self, input_data):
        layer_count = 0
        for layer in self.deepreg_layers:
            if layer_count == 0: #--- the input layer
                output_data = layer(input_data)
            else:
                output_data = layer(output_data)
            layer_count += 1
        return output_data