import torch.nn as nn

############## Define Deep Regression Model ##################
class DeepReg(nn.Module):
    def __init__(self, layer_num, input_dim, hidden_dim, output_dim):
        super(DeepReg, self).__init__()

        kern_size = 17
        stride_size = 17
        if input_dim==80: #Account for NPA
            kern_size-=1
            stride_size-=1
        
        self.deepreg_layers = nn.ModuleList()
        cur_layer = nn.Conv1d(in_channels=input_dim,out_channels=input_dim,kernel_size=kern_size,stride=stride_size,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(5))
        self.deepreg_layers.append(nn.LeakyReLU())
        #cur_layer = nn.GRU(input_size=5,hidden_size=50,bias=True)#out_features=50,bias=True)
        cur_layer = nn.AdaptiveAvgPool1d(50)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(50))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=50,out_features=25,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(25))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=25,out_features=10,bias=True)
        self.deepreg_layers.append(cur_layer)
        self.deepreg_layers.append(nn.BatchNorm1d(10))
        self.deepreg_layers.append(nn.LeakyReLU())
        cur_layer = nn.Linear(in_features=10,out_features=1,bias=True)
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