%% 
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
rate_hz = [10,15,7.5,5]; % frame rate of each video
radius = [5,6,6,6];
sub_added = 'add_neurons_0.005_rotate';

data_ind = 2;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\data_CNMFE',[data_name,'_original_masks'],sub_added);
% path_name = fullfile('E:\data_CNMFE',[data_name,'_added_blockwise_weighted_sum_unmask'],sub_added);
list_Exp_ID = cellfun(@(x) [data_name,x,'_added'], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_
dir_GT_info = fullfile(path_name,'GT info'); % FinalMasks_

dir_CNMFE_eval = 'C:\Other methods\CNMF_E-1.1.2';
CNMFE_date = '20220925';
dir_min1pipe_eval = 'C:\Other methods\MIN1PIPE-3.0.0';
min1pipe_date = '20220922';
gSiz = 2*radius(data_ind);
[all_recall_add,all_recall_all,all_precision_all,all_F1_all] = deal(cell(1,4));

%% MIN1PIPE
load(fullfile(dir_min1pipe_eval,['eval_',data_name,'_',sub_added,'_thb history ',min1pipe_date,'.mat']),...
    'best_Recall','best_Precision','best_F1','best_time','best_ind_param',...
    'best_thb','ind_param','list_params','history','best_history'); % _added_blockwise_
end_history = best_history(end,:);
best_param = end_history(1:end-5);
pix_select_sigthres = best_param(1);
pix_select_corrthres = best_param(2);
merge_roi_corrthres = best_param(3);
dt = best_param(4);
kappa = best_param(5);
se = best_param(6);
dir_sub = sprintf('min1pipe\\pss=%0.2f_psc=%0.2f_mrc=%0.2f_dt=%0.2f_kappa=%0.2f_se=%d',...
    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa, se);

dir_output_masks=fullfile(path_name,dir_sub); % online
[list_recall_add,list_recall_all,list_precision_all,list_F1_all] = deal(zeros(num_Exp,1));
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_output_masks,[Exp_ID,'_Masks_',num2str(best_thb),'.mat']),'Masks3');
%     Masks = permute(Masks,[3,2,1]);
    load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'masks_add','Ysiz');
    Masks2 = reshape(Masks3, Ysiz(1)*Ysiz(2),[]);
    masks_add2 = reshape(masks_add, Ysiz(1)*Ysiz(2),[]);
    FinalMasks2 = reshape(FinalMasks, Ysiz(1)*Ysiz(2),[]);
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add2,Masks2,0.5);
    list_recall_add(eid) = Recall;
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(FinalMasks2,Masks2,0.5);
    list_recall_all(eid) = Recall;
    list_precision_all(eid) = Precision;
    list_F1_all(eid) = F1;
end
all_recall_add{1} = list_recall_add;
all_recall_all{1} = list_recall_all;
all_precision_all{1} = list_precision_all;
all_F1_all{1} = list_F1_all;

%% CNMF-E
load(fullfile(dir_CNMFE_eval,['eval_',data_name,'_',sub_added,'_thb history ',CNMFE_date,'.mat']),...
    'best_Recall','best_Precision','best_F1','best_time','best_ind_param',...
    'best_thb','ind_param','list_params','history','best_history'); % _added_blockwise_
end_history = best_history(end,:);
best_param = end_history(1:end-5);
rbg = best_param(1);
nk = best_param(2);
rdmin = best_param(3);
min_corr = best_param(4);
min_pnr = best_param(5);
merge_thr = best_param(6);
mts = best_param(7);
mtt = best_param(8);
dir_sub = sprintf('CNMFE\\gSiz=%d,rbg=%0.1f,nk=%d,rdmin=%0.1f,mc=%0.2f,mp=%d,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
    gSiz,rbg,nk,rdmin,min_corr,min_pnr,merge_thr,mts,mtt);

dir_output_masks=fullfile(path_name,dir_sub); % online
[list_recall_add,list_recall_all,list_precision_all,list_F1_all] = deal(zeros(num_Exp,1));
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_output_masks,[Exp_ID,'_Masks_',num2str(best_thb),'.mat']),'Masks3');
%     Masks = permute(Masks,[3,2,1]);
    load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'masks_add','Ysiz');
    Masks2 = reshape(Masks3, Ysiz(1)*Ysiz(2),[]);
    masks_add2 = reshape(masks_add, Ysiz(1)*Ysiz(2),[]);
    FinalMasks2 = reshape(FinalMasks, Ysiz(1)*Ysiz(2),[]);
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add2,Masks2,0.5);
    list_recall_add(eid) = Recall;
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(FinalMasks2,Masks2,0.5);
    list_recall_all(eid) = Recall;
    list_precision_all(eid) = Precision;
    list_F1_all(eid) = F1;
end
all_recall_add{2} = list_recall_add;
all_recall_all{2} = list_recall_all;
all_precision_all{2} = list_precision_all;
all_F1_all{2} = list_F1_all;

%% SUNS FISSA
dir_sub='\complete_FISSA\4816[1]th2'; %_2out+BGlayer
dir_output_masks=fullfile(path_name,dir_sub,'output_masks'); % online
[list_recall_add,list_recall_all,list_precision_all,list_F1_all] = deal(zeros(num_Exp,1));
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_output_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
    Masks = permute(Masks,[3,2,1]);
    load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'masks_add','Ysiz');
    Masks2 = reshape(Masks, Ysiz(1)*Ysiz(2),[]);
    masks_add2 = reshape(masks_add, Ysiz(1)*Ysiz(2),[]);
    FinalMasks2 = reshape(FinalMasks, Ysiz(1)*Ysiz(2),[]);
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add2,Masks2,0.5);
    list_recall_add(eid) = Recall;
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(FinalMasks2,Masks2,0.5);
    list_recall_all(eid) = Recall;
    list_precision_all(eid) = Precision;
    list_F1_all(eid) = F1;
end
all_recall_add{3} = list_recall_add;
all_recall_all{3} = list_recall_all;
all_precision_all{3} = list_precision_all;
all_F1_all{3} = list_F1_all;

%% SUNS TUnCaT
dir_sub='\complete_TUnCaT\4816[1]th4';
dir_output_masks=fullfile(path_name,dir_sub,'output_masks'); % online
[list_recall_add,list_recall_all,list_precision_all,list_F1_all] = deal(zeros(num_Exp,1));
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_output_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
    Masks = permute(Masks,[3,2,1]);
    load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'masks_add','Ysiz');
    Masks2 = reshape(Masks, Ysiz(1)*Ysiz(2),[]);
    masks_add2 = reshape(masks_add, Ysiz(1)*Ysiz(2),[]);
    FinalMasks2 = reshape(FinalMasks, Ysiz(1)*Ysiz(2),[]);
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add2,Masks2,0.5);
    list_recall_add(eid) = Recall;
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(FinalMasks2,Masks2,0.5);
    list_recall_all(eid) = Recall;
    list_precision_all(eid) = Precision;
    list_F1_all(eid) = F1;
end
all_recall_add{4} = list_recall_add;
all_recall_all{4} = list_recall_all;
all_precision_all{4} = list_precision_all;
all_F1_all{4} = list_F1_all;

%%
Table_recall_add = cell2mat(all_recall_add);
Table_recall_all = cell2mat(all_recall_all);
Table_precision_all = cell2mat(all_precision_all);
Table_F1_all = cell2mat(all_F1_all);
Table_recall_add_ext=[Table_recall_add;nanmean(Table_recall_add,1);nanstd(Table_recall_add,1,1)];
Table_recall_all_ext=[Table_recall_all;nanmean(Table_recall_all,1);nanstd(Table_recall_all,1,1)];
Table_precision_all_ext=[Table_precision_all;nanmean(Table_precision_all,1);nanstd(Table_precision_all,1,1)];
Table_F1_all_ext=[Table_F1_all;nanmean(Table_F1_all,1);nanstd(Table_F1_all,1,1)];
disp(Table_recall_add_ext(end-1,:))
Table_all = [Table_recall_add_ext,Table_recall_all_ext,Table_precision_all_ext,Table_F1_all_ext];
