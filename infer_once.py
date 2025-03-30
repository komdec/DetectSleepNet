import glob
import os
import torch
import numpy as np
from logger import get_logger
from torch.utils.data import DataLoader
from dataset import SleepDataset
from sklearn import metrics
from tqdm import tqdm
from model import *
from utils import *

def run(model_weights, flavor_model, dataset_name, idx_path, dataset_dir=None, \
            classes=5, try_gpu=True,):

    # Config file
    # Output directory
    output_dir = f'output/{flavor_model}_{dataset_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger, file_handler, console_handler = get_logger(os.path.join(output_dir, f'train.log'), level="info")
    logger.info("###################################") 
    logger.info("***********************************") 
    logger.info("flavor_model : " + str(flavor_model))
    if try_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    else:
        device="cpu"
    logger.info("Using device: {}".format(device))

    net = eval(flavor_model)(num_classes=classes)
    total = sum([param.nelement() for param in net.parameters()])
    
    if total<1e6:
        logger.info("Number of parameter: %.2fK" % (total/1e3))
    else:
        logger.info("Number of parameter: %.2fM" % (total/1e6))


    # output_dir_train=os.path.join(output_dir, 'train')
    # output_fold_dir = os.path.join(output_dir_train, str(0))

    # if not os.path.exists(output_fold_dir): 
    #     os.makedirs(output_fold_dir)  

    if dataset_name == "phy2018":
        model_weights_list = []
        for i in range(5):
            model_weights_list.append(model_weights+'/'+str(i)+'/model_checkpoint.pth')
    elif dataset_name == "shhs1":
        model_weights_list = [model_weights]
    idx_file = np.load(idx_path,allow_pickle=True)
    folds = len(model_weights_list)
    np.set_printoptions(threshold=50)
    test_acc_fine, test_f1_classes_fine, test_f1w_fine, test_f1m_fine, test_k_fine = [], [], [], [], []
    for fold_idx in range(folds):
        train_idx = idx_file[fold_idx]['train']
        valid_idx = idx_file[fold_idx]['val']
        test_idx = idx_file[fold_idx]['test']
        if folds>1:
            logger.info("******    fold:  {}/{}   *******".format(fold_idx,folds))    
        logger.info("*******    dataset info    ********")
        logger.info("Dataset name : {}".format(dataset_name))
        logger.info("Train data idx : {}".format(train_idx))
        logger.info("Valid data idx : {}".format(valid_idx))
        logger.info("Test data idx : {}".format(test_idx))
        test_dataset = SleepDataset(dataset_dir, test_idx, dataset_name)
        testloader = DataLoader(test_dataset,batch_size=1)
        model_dict = torch.load(model_weights_list[fold_idx])
        net.load_state_dict(model_dict,strict=False)
        net = net.to(device)
        net.eval()
        test_acc_fold, test_k_fold, test_f1_classes_fold, test_f1w_fold, test_f1m_fold = [], [], [], [], []
        # =====================test============================
        with torch.no_grad():
            y_list = []
            pred_list = []    
            for idx,(x,y) in enumerate(tqdm(testloader)):
                x = torch.transpose(x,1,2).to(device=device)
                y = y.to(device=device)
                outputs= net(x)
                outputs = outputs.view(-1, outputs.shape[-1])
                y = y.view(-1)
                # pred = F.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)
                y_list.append(y.detach().cpu()) 
                pred_list.append(pred.detach().cpu())
        # metrics
        y_list = torch.cat(y_list).numpy()
        pred_list = torch.cat(pred_list).numpy()
        test_acc_fold = (y_list==pred_list).mean().item()
        test_k_fold = metrics.cohen_kappa_score(y_list, pred_list)
        test_f1_classes_fold = metrics.f1_score(y_list,pred_list, average=None)
        test_f1w_fold = metrics.f1_score(y_list,pred_list, average='weighted')
        test_f1m_fold = metrics.f1_score(y_list,pred_list, average='macro')
        if folds>1:
            logger.info('fold:{}, test_acc:{}, test_f1_classes:{}, test_f1_weighted:{}, test_f1_macro:{}, test_k:{}'.format(fold_idx, test_acc_fine, test_f1_classes_fine, test_f1w_fine, test_f1m_fine, test_k_fine))
    test_acc_fine.append(np.average(test_acc_fold))
    test_f1_classes_fine.append(test_f1_classes_fold)
    test_f1w_fine.append(np.average(test_f1w_fold))
    test_f1m_fine.append(np.average(test_f1m_fold))
    test_k_fine.append(np.average(test_k_fold))

    np_test_acc = np.around(np.array(test_acc_fine)*100, 3)
    test_f1_classes_fine = np.around(np.average(np.array(test_f1_classes_fine),0), 2)
    np_test_f1w = np.around(np.array(test_f1w_fine), 3)
    np_test_f1m = np.around(np.array(test_f1m_fine), 3)
    np_test_k = np.around(np.array(test_k_fine), 3)

    logger.info("***********************************") 
    if total<1e6:
        logger.info("Number of parameter: %.2fK" % (total/1e3))
    else:
        logger.info("Number of parameter: %.2fM" % (total/1e6))
    logger.info("flavor_model : " + str(flavor_model))
    if folds>1:
        logger.info("np_test_a_list : " + str(np_test_acc))
    logger.info("avg_test_acc : " + str(np.average(np_test_acc)))
    logger.info("avg_test_f1_weight : " + str(np.average(np_test_f1w)))
    logger.info("avg_test_f1_macro : " + str(np.average(np_test_f1m)))
    logger.info("avg_test_kappa : " + str(np.average(np_test_k)))
    logger.info("avg_test_f1_classes : " + str((test_f1_classes_fine)))
    logger.info("***********************************") 
    logger.info("###################################") 
    logger.info("  ") 
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


