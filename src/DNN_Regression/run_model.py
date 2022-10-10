import torch
from model_files.util import *
from model_files.nn_model_cnn import *
from tqdm import tqdm
import pandas as pd
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import random
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from torchsample.modules import ModuleTrainer
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/submit/aidandc/miniconda3/lib/

#plt.style.use("seaborn")
#plt.style.use("ggplot")
#plt.style.use("bmh")
plt.style.use("seaborn-darkgrid")
#plt.style.use("seaborn-deep")
#plt.style.use("tableau-colorblind10")

directory_name = input('Directory name for training results: ')
if len(directory_name)==0:
    print('Improper directory name')
    quit()
data_type = input('Data type (original/extended/extended_npa): ')
if data_type!='original' and data_type!='extended' and data_type!='extended_npa':
    print('Improper data type selected')
    quit()
reg_value = float(input('L1 regularization value (0 for no regularization): '))    
val_patience = int(input('Patience for early-stopping using validation loss (-1 for no overfitting): '))

def train_deepreg(trn_file, val_file, tst_file, 
                  normalization_type, gpu_id, 
                  trn_batch_size, val_batch_size, tst_batch_size,
                  layer_num, input_dim, hidden_dim, output_dim,learning_rate,max_epoch,check_freq):
    
    # set up folders to have training results
    dir_name = str(directory_name+'/training_results/')
    dir_model_name = str(directory_name+'/training_models')
    dir_name_test = str(directory_name+'/test_results/')
    make_dir([dir_name, dir_model_name, dir_name_test])
    
    # set up device
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    
    # load training set, validation set, and held-out (test) set
    scaler_save_file_name = os.path.join(dir_model_name, '{}_scaler.gz'.format(normalization_type))
    scaler = generate_scaler(trn_file, scaler_save_file_name, normalization_type)
    
    #--- process training data, val data, test
    trn_features, trn_targets = get_normalized_dataset(trn_file, scaler)
    val_features, val_targets = get_normalized_dataset(val_file, scaler)
    tst_features, tst_targets = get_normalized_dataset(tst_file, scaler)
    
    #--- put data into dataloader
    trn_num = len(trn_targets)
    val_num = len(val_targets)
    tst_num = len(tst_targets)
    trn_loader = prepare_data(trn_features, trn_targets, trn_batch_size, trn_num)
    val_loader = prepare_data(val_features, val_targets, val_batch_size, val_num)
    tst_loader = prepare_data(tst_features, tst_targets, tst_batch_size, tst_num)
    
    #--- create model
    deep_reg_Net = DeepReg(layer_num, input_dim, hidden_dim, output_dim)
    print(deep_reg_Net)
    
    deep_reg_Net.to(device)
    
    
    #--- optimizer
    optimizer = torch.optim.Adam(params=deep_reg_Net.parameters(), lr=learning_rate)
    
    #--- loss fuction
    l2_loss_func = L2_Func()
    
    
    #---- start to train our network
    total_trn_loss = []
    total_trn_RMSE = []
    total_val_RMSE = []
    total_epoch = []

    train_RMSE = []
    val_RMSE = []

    best_RMSE = float('inf')
    best_epoch = 0
    val_increases = 0
    
    for epoch in range(max_epoch):
        print('')
        print('')
        print('###################### Start to Train NN model ##########################')
        deep_reg_Net.train()
        epoch_loss = []
        progress = tqdm(total=len(trn_loader), desc='epoch % 3d' % epoch)
        for step, (X_features, Y_targets, idx) in enumerate(trn_loader):
            # zero the parameter gradients
            optimizer.zero_grad()
            
        
            ################## Get Training & Traget Dataset ##################
            X_features = X_features.to(device).float()
            Y_targets = Y_targets.to(device).flatten().float()
            
            # forward + backward + optimize
            Y_prediction = deep_reg_Net(X_features).flatten().float()
            loss = l2_loss_func(Y_prediction, Y_targets)  # MSE loss
            loss = torch.sqrt(loss) #RMSE

            #Optional L1 Regularization, comment out if not wanted
            l1_lambda = reg_value #Scalable parameter for penalization for large weights
            l1_norm = sum(p.abs().sum() for p in deep_reg_Net.parameters())
            loss = loss + l1_norm * l1_lambda

            loss.backward()
            optimizer.step()
            
            #---finished update in one batch
            epoch_loss.append(loss.data.cpu().numpy())
            #progress.set_postfix({'loss': loss.data.cpu().numpy()})
            progress.update()
        progress.close()
        total_trn_loss.append(np.mean(epoch_loss))  #---- finished one epoch
        
        
        #------ validation our model
        if epoch%check_freq==0:
            trn_rmse, gt_pred_dict = val_deepreg(deep_reg_Net, trn_loader, device, l2_loss_func)
            total_trn_RMSE.append(trn_rmse)
            
            val_rmse, gt_pred_dict = val_deepreg(deep_reg_Net, val_loader, device, l2_loss_func)
            total_val_RMSE.append(val_rmse)

            if best_RMSE > val_rmse:
                best_RMSE = val_rmse
                best_epoch = epoch
                ################ check and always save the best model we have
                model_file_name = os.path.join(dir_model_name, 'best_net_L{}_H{}.pt'.format(layer_num, hidden_dim))
                save_model(deep_reg_Net.eval(), model_file_name)
            figure_name = os.path.join(dir_name, 'train_val_mse_L{}_H{}.png'.format(layer_num, hidden_dim))
            display_RMSE(total_trn_RMSE, total_val_RMSE, check_freq, figure_name)
            #figure_name = os.path.join(dir_name, 'train_loss_L{}_H{}.png'.format(layer_num, hidden_dim))
            #display_train_loss(total_trn_loss, figure_name)
            if epoch>0 and val_patience>=0:
                if total_val_RMSE[epoch]>total_val_RMSE[epoch-1] and total_trn_RMSE[epoch]<total_trn_RMSE[epoch-1]:
                    val_increases+=1
        if val_increases>=val_patience and val_patience>=0:
            print('Early stopping after '+str(val_increases)+' increases to val loss')
            break

            
           
    ###### after training, let's get the best and verify its performance on training/validation/test set again!
    # load the best net
    model_file_name = os.path.join(dir_model_name, 'best_net_L{}_H{}.pt'.format(layer_num, hidden_dim))
    best_Net = DeepReg(layer_num, input_dim, hidden_dim, output_dim)
    if torch.cuda.is_available():
        try:
            best_Net.load_state_dict(torch.load(model_file_name, map_location='cuda:{}'.format(gpu_id)))
            print('Loading Pretrained models 1 (GPU)!')
        except:
            best_Net = nn.DataParallel(best_Net)
            best_Net.load_state_dict(torch.load(model_file_name, map_location='cuda:{}'.format(gpu_id)))
            print('Loading Pretrained models 2 (GPU)!')
    else:
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        best_Net.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu')))
        print('Loading Pretrained models on CPU!')
    
    # move the best net to our device
    best_Net.to(device)
    
        
        
    # check its performance on training set
    trn_rmse, trn_gt_pred_dict = val_deepreg(best_Net, trn_loader, device, l2_loss_func)
    
    # check its performance on validation set
    val_rmse, val_gt_pred_dict = val_deepreg(best_Net, val_loader, device, l2_loss_func)
    
    # check its performance on test set
    tst_rmse, tst_gt_pred_dict = val_deepreg(best_Net, tst_loader, device, l2_loss_func)
    
    
    #----------- Finally, let's save our results---------------#
    trn_pred_file = os.path.join(dir_name_test, 'train_prediction_L{}_H{}.csv'.format(layer_num, hidden_dim))
    trn_df = pd.DataFrame.from_dict(trn_gt_pred_dict)
    trn_df.to_csv(trn_pred_file, index=False)
    
    val_pred_file = os.path.join(dir_name_test, 'validation_prediction_L{}_H{}.csv'.format(layer_num, hidden_dim))
    val_df = pd.DataFrame.from_dict(val_gt_pred_dict)
    val_df.to_csv(val_pred_file, index=False)
    
    tst_pred_file = os.path.join(dir_name_test, 'hos_test_prediction_L{}_H{}.csv'.format(layer_num, hidden_dim))
    tst_df = pd.DataFrame.from_dict(tst_gt_pred_dict)
    tst_df.to_csv(tst_pred_file, index=False)
    
    final_RMSE_dict = {'Type':['Train_RMSE', 'Validation_RMSE', 'Test_RMSE'],
                       'RMSE':[trn_rmse, val_rmse, tst_rmse]}
    
    final_RMSE_df = pd.DataFrame.from_dict(final_RMSE_dict)
    final_RMSE_file = os.path.join(dir_name_test, 'final_RMSE_L{}_H{}.csv'.format(layer_num, hidden_dim))
    final_RMSE_df.to_csv(final_RMSE_file, index=False)
    
    print('')
    print('')
    print('>>>Congrats! The DNN regression model has been trained and saved!')

#------ define function for validation------#
def val_deepreg(model, data_loader, device, l2_loss_func):
    model.eval()
    RMSE = []
    gt_pred_dict = {'Y_True':[], 'Y_Prediction':[]}
    with torch.no_grad():
        for step, (X_features, Y_targets, idx) in enumerate(data_loader):
            X_features = X_features.to(device).float()
            Y_targets = Y_targets.to(device).flatten().float()
            Y_prediction = model(X_features).flatten().float().detach()
            cur_mse = l2_loss_func(Y_prediction, Y_targets)
            cur_mse = torch.sqrt(cur_mse) #RMSE
            RMSE.append(cur_mse.item())
            ##### let's save our ground truth and predictions
            Y_truth_list = list(Y_targets.cpu().numpy().flatten())
            Y_prediction_list = list(Y_prediction.cpu().numpy().flatten())
            gt_pred_dict['Y_True'].extend(Y_truth_list)
            gt_pred_dict['Y_Prediction'].extend(Y_prediction_list)
    avg_rmse = np.mean(RMSE)
    return avg_rmse, gt_pred_dict

if __name__ == '__main__':
    
    # Set random seed for reproducibility
    #manualSeed = 9708
    manualSeed = random.randint(1, 10000) # use this line if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    ################ Parameters Settings ######################
    trn_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/dnn_dataset/training.csv'
    val_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/dnn_dataset/validation.csv'
    tst_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/dnn_dataset/hos.csv'
    if data_type=='extended':
        trn_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/extended_dataset/training_extended.csv'
        val_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/extended_dataset/validation_extended.csv'
        tst_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/extended_dataset/hos_extended.csv'
    if data_type=='extended_npa':
        trn_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/extended_dataset_npa/training_extended_npa.csv'
        val_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/extended_dataset_npa/validation_extended_npa.csv'
        tst_file = '/home/submit/aidandc/SuperCDMS/pytorch/data/extended_dataset_npa/hos_extended_npa.csv'    
    normalization_type = 'StandardScaler'
    gpu_id = 0
    trn_batch_size = 80#128
    val_batch_size = 80#128
    tst_batch_size = 80#128
    layer_num = 10
    input_dim = 19
    if data_type=='extended':
        input_dim = 85
    if data_type=='extended_npa':
        input_dim = 80
    hidden_dim = 32
    output_dim = 1
    learning_rate = 1e-3
    
    max_epoch = 30
    check_freq = 1
    
    
    ################ Start Training ######################
    train_deepreg(trn_file,
                  val_file, 
                  tst_file, 
                  normalization_type, 
                  gpu_id, 
                  trn_batch_size, 
                  val_batch_size, 
                  tst_batch_size,
                  layer_num, 
                  input_dim, 
                  hidden_dim, 
                  output_dim,
                  learning_rate,
                  max_epoch,
                  check_freq)
    print("Random Seed: ", manualSeed)
