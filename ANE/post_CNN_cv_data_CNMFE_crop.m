% data_ind = 1; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th5');
% data_ind = 2; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th4');
% data_ind = 3; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th4');
% data_ind = 4; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF50','4816[1]th4');
% data_ind = 1; dir_SUNS_sub = fullfile('SUNS_FISSA_SF25','4816[1]th3');
% data_ind = 2; dir_SUNS_sub = fullfile('SUNS_FISSA_SF25','4816[1]th2');
% data_ind = 3; dir_SUNS_sub = fullfile('SUNS_FISSA_SF25','4816[1]th2');
% data_ind = 4; dir_SUNS_sub = fullfile('SUNS_FISSA_SF50','4816[1]th2');
addpath(genpath('.'))

%%
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_lam = [15,5,8,8];
% list_th_SNR = [5,4,4,4];

data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false); % ,'_added'
num_Exp = length(list_Exp_ID);

dir_parent=fullfile('..','data','data_CNMFE',data_name);
d0 = 0.8;
lam = list_lam(data_ind);
dir_GT_info = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));

dir_SUNS = fullfile(dir_parent, dir_SUNS_sub);
dir_masks = fullfile(dir_SUNS, 'output_masks');
sub_folder = 'add_new_blockwise';
dir_add_new = fullfile(dir_masks, sub_folder);
dir_GT = fullfile(dir_parent, 'GT Masks');

load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
mean([list_Recall,list_Precision,list_F1]);
list_time_SUNS = list_time(:,end);
[list_Recall, list_Precision, list_F1, list_Recall_add, list_Precision_add, ...
    list_Recall_add_max, list_Recall_max, list_Precision_max, list_F1_max, list_time_weights, ...
    list_time_classifier, list_time_merge] = deal(zeros(num_Exp,1));

%% merge repeated neurons in list_added
folder = sprintf('trained dropout %gexp(-%g)',d0,lam);
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
    load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'masks_added_full','masks_added_crop','images_added_crop','time_weights'); % ,'list_valid'
    list_time_weights(eid) = time_weights;
    try 
        load(fullfile(dir_add_new,folder,['CNN_predict_',Exp_ID,'_cv',num2str(eid-1),'.mat']), 'pred_valid','time_CNN'); 
        list_time_classifier(eid) = time_CNN;

        tic;
        masks=permute(logical(Masks),[3,2,1]);
        list_added_all = masks_added_full(:,:,pred_valid);
        [Lx,Ly,num_added] = size(list_added_all);
        list_added_sparse = sparse(reshape(list_added_all,Lx*Ly,num_added));
        times = cell(1,num_added);
        [list_added_sparse_half,times] = piece_neurons_IOU(list_added_sparse,0.5,0.5,times);
        [list_added_sparse_final,times] = piece_neurons_consume(list_added_sparse_half,inf,0.5,0.75,times);
        list_added_final = reshape(full(list_added_sparse_final),Lx,Ly,[]);

        %%
        Masks = cat(3,masks,list_added_final);
        list_time_merge(eid) = toc;
    catch
        masks=permute(logical(Masks),[3,2,1]);
        Masks=masks;
        [Lx, Ly, ~] = size(Masks);
    end
    save(fullfile(dir_add_new,folder,['Output_Masks_',Exp_ID,'_added.mat']),'Masks');
    Masks_2 = sparse(reshape(Masks,Lx*Ly,[]));
    n_init = size(masks,3);
    n_add = size(list_added_final,3);

    %%
    load(fullfile(dir_GT_info,['DroppedMasks_',Exp_ID,'.mat']),'DroppedMasks');
    masks_add_2 = sparse(reshape(DroppedMasks,Lx*Ly,[]));
    load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
    [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,0.5);
    list_Recall(eid) = Recall;
    list_Precision(eid) = Precision;
    list_F1(eid) = F1;
    list_Precision_add(eid) = sum(m(:,n_init+1:end),'all')/n_add;
    [Recall_add, ~, ~, ~] = GetPerformance_Jaccard_2(masks_add_2,Masks_2,0.5);
    list_Recall_add(eid) = Recall_add;
end
%%
list_time = [list_time_SUNS, list_time_weights, list_time_classifier, list_time_merge];
list_time = [list_time, sum(list_time,2)];
Table = [list_Recall_add,list_Precision_add,list_Recall,list_Precision,list_F1,list_time(:,end)];
Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
disp(Table_ext(1:end-1,3:end));
save(fullfile(dir_add_new,folder,'eval.mat'),'list_Recall_add',...
    'list_Precision_add','list_Recall','list_Precision','list_F1','list_time');
% disp('Finished this step');
