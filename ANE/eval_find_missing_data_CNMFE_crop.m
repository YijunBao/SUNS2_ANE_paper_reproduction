%%
% folder of the GT Masks
dir_parent='E:\data_CNMFE\';
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};

data_ind = 3;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);

dir_parent=fullfile('E:\data_CNMFE\',data_name);
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_added_auto']);
dir_GT = fullfile(dir_parent, 'GT Masks');
% dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_auto'], 'GT Masks');
dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT_SF25\4816[1]th4'); % 4 v1
dir_masks = fullfile(dir_SUNS, 'output_masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');
dir_drop = fullfile(dir_add_new, 'trained dropout 0.8exp(-10)');
dir_eval = fullfile(dir_drop,'\avg_Xmask_0.5\classifier_res0_0+2 frames');

% load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
% mean([list_Recall,list_Precision,list_F1])
% list_time_SUNS = list_time(:,end);
% [list_Recall, list_Precision, list_F1, list_time_weights, ...
%     list_time_classifier, list_time_merge] = deal(zeros(num_Exp,1));

%% merge repeated neurons in list_added
% for eid = 1:num_Exp
%     Exp_ID = list_Exp_ID{eid};
%     load(fullfile(dir_add_new,['Output_Masks_',Exp_ID,'_added.mat']),'Masks');
%     [Lx,Ly,n] = size(Masks);
%     Masks_2 = sparse(reshape(Masks,Lx*Ly,[]));
%     
%     %%
%     load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
%     [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,0.5);
%     list_Recall(eid) = Recall;
%     list_Precision(eid) = Precision;
%     list_F1(eid) = F1;
% end
%%
load(fullfile(dir_eval,'eval.mat'),'list_time','list_Recall_add','list_Recall','list_Precision','list_F1');
Table = [list_Recall,list_Precision,list_F1,list_time(:,end)];
Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
disp(nanmean(Table,1));
