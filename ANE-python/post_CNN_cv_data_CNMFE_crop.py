import numpy as np
import sys
import os
import scipy.io as sio
import time
from scipy import sparse

sys.path.insert(1, '../SUNS2') # the path containing "suns" folder
from suns.PostProcessing.combine import piece_neurons_IOU, piece_neurons_consume
from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2

# data_ind = 1; dir_masks = os.path.join('SUNS_TUnCaT_SF25','4816[1]th5','output_masks');
# data_ind = 2; dir_masks = os.path.join('SUNS_TUnCaT_SF25','4816[1]th4','output_masks');
# data_ind = 3; dir_masks = os.path.join('SUNS_TUnCaT_SF25','4816[1]th4','output_masks');
# data_ind = 1; dir_masks = os.path.join('SUNS_FISSA_SF25','4816[1]th3','output_masks');
# data_ind = 3; dir_masks = os.path.join('SUNS_TUnCaT_SF50','4816[1]th4','output_masks');
# data_ind = 2; dir_masks = os.path.join('SUNS_FISSA_SF25','4816[1]th2','output_masks');
# data_ind = 3; dir_masks = os.path.join('SUNS_FISSA_SF25','4816[1]th2','output_masks');
# data_ind = 4; dir_masks = os.path.join('SUNS_FISSA_SF50','4816[1]th2','output_masks');
##
if __name__ == "__main__":
    dir_masks = sys.argv[1] # "SUNS_TUnCaT_SF50/4816[1]th4/output_masks" # 
    data_ind = int(sys.argv[2]) # 3 # 

    list_data_names = ['blood_vessel_10Hz', 'PFC4_15Hz', 'bma22_epm', 'CaMKII_120_TMT Exposure_5fps']
    list_ID_part = ['_part11', '_part12', '_part21', '_part22']
    list_lam = [15,5,8,8]
    # list_th_SNR = [5,4,4,4]

    data_name = list_data_names[data_ind]
    list_Exp_ID = [data_name+x for x in list_ID_part]

    num_Exp = len(list_Exp_ID)
    dir_parent = os.path.join('..','data','data_CNMFE',data_name)
    d0 = 0.8
    lam = list_lam[data_ind]
    dir_GT_info = os.path.join(dir_parent,'GT Masks dropout {}exp(-{})'.format(d0, lam))
    dir_masks = os.path.join(dir_parent,dir_masks)
    sub_folder = 'add_new_blockwise'
    dir_add_new = os.path.join(dir_masks,sub_folder)
    dir_GT = os.path.join(dir_parent,'GT Masks')
    Output_Info_All = sio.loadmat(os.path.join(dir_masks,'Output_Info_All.mat'))
    list_time_SUNS = Output_Info_All['list_time'][:,-1]
    list_Recall = np.zeros(num_Exp)
    list_Precision = np.zeros(num_Exp)
    list_F1 = np.zeros(num_Exp)
    list_time_weights = np.zeros(num_Exp)
    list_time_classifier = np.zeros(num_Exp)
    list_time_merge = np.zeros(num_Exp)
    
    ## merge repeated neurons in list_added
    folder = 'trained dropout {}exp(-{})'.format(d0, lam)
    for eid in range(num_Exp):
        Exp_ID = list_Exp_ID[eid]
        Output_Masks = sio.loadmat(os.path.join(dir_masks,'Output_Masks_'+Exp_ID+'.mat'))
        Masks = Output_Masks['Masks'].astype('bool')
        added_auto_blockwise = sio.loadmat(os.path.join(dir_add_new,Exp_ID+'_added_auto_blockwise.mat'))
        masks_added_full = added_auto_blockwise['masks_added_full'].astype('bool')
        masks_added_crop = added_auto_blockwise['masks_added_crop'].astype('bool')
        images_added_crop = added_auto_blockwise['images_added_crop']
        time_weights = added_auto_blockwise['time_weights'].squeeze()
        list_time_weights[eid] = time_weights
        try:
            CNN_predict = sio.loadmat(os.path.join(dir_add_new,folder,'CNN_predict_{}_cv{}.mat'.format(Exp_ID,eid)))
            pred_valid = CNN_predict['pred_valid'].squeeze().astype('bool')
            time_CNN = CNN_predict['time_CNN'].squeeze()
            list_time_classifier[eid] = time_CNN
            tic = time.time()
            masks = Masks#.transpose([2,1,0])
            list_added_all = masks_added_full[pred_valid,:,:]
            num_added,Ly,Lx = list_added_all.shape
            list_added_sparse = sparse.csr_matrix(list_added_all.reshape((num_added,Ly * Lx)))
            # times = np.arange(num_added)
            times = [[] for _ in range(num_added)]
            list_added_sparse_half,times = piece_neurons_IOU(list_added_sparse,0.5,0.5,times)
            list_added_sparse_final,times = piece_neurons_consume(list_added_sparse_half,np.inf,0.5,0.75,times)
            list_added_final = list_added_sparse_final.A.reshape((-1,Ly,Lx)).astype('bool')
            ##
            Masks = np.concatenate([masks,list_added_final],0)
            toc = time.time()
            list_time_merge[eid] = toc - tic
        except:
            print('Cannot find the ANE output of {}. Use the SUNS output as the final result.'.format(Exp_ID))
        
        sio.savemat(os.path.join(dir_add_new,folder,'Output_Masks_'+Exp_ID+'_added.mat'),\
            {'Masks':Masks.transpose([2,1,0])}, do_compression=True)
        _, Ly, Lx = Masks.shape
        Masks_2 = sparse.csr_matrix(Masks.reshape(-1, Ly * Lx))
        # n_init = masks.shape[0]
        # n_add = list_added_final.shape[0]
        ##
        # DroppedMasks = sio.loadmat(os.path.join(dir_GT_info,'DroppedMasks_'+Exp_ID+'.mat'))
        # DroppedMasks = DroppedMasks['DroppedMasks'].transpose([2,1,0]).astype('bool')
        # masks_add_2 = sparse.csr_matrix(DroppedMasks.reshape(-1, Ly * Lx))
        GTMasks_2 = sio.loadmat(os.path.join(dir_GT,'FinalMasks_'+Exp_ID+'_sparse.mat'))
        GTMasks_2 = GTMasks_2['GTMasks_2'].astype('bool').T
        Recall,Precision,F1 = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,0.5)
        list_Recall[eid] = Recall
        list_Precision[eid] = Precision
        list_F1[eid] = F1

    ##
    list_time = np.stack([list_time_SUNS,list_time_weights,list_time_classifier,list_time_merge], 1)
    list_time = np.concatenate([list_time,list_time.sum(1, keepdims=True)], 1)
    Table = np.stack([list_Recall,list_Precision,list_F1,list_time[:,-1]], 1)
    Table_ext = np.concatenate([Table, np.nanmean(Table,0, keepdims=True), np.nanstd(Table, 0, ddof=1, keepdims=True)], 0)
    print(Table_ext[:-1, :])
    sio.savemat(os.path.join(dir_add_new,folder,'eval.mat'),\
        {'list_Recall':list_Recall,'list_Precision':list_Precision,
            'list_F1':list_F1,'list_time':list_time,'Table_ext':Table_ext})
