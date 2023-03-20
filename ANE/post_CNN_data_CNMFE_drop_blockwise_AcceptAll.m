addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
% folder of the GT Masks
% dir_parent='E:\data_CNMFE\';
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};

data_ind = 3;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false); % ,'_added'
num_Exp = length(list_Exp_ID);

% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
dir_parent=fullfile('E:\data_CNMFE\',data_name);
d0 = 0.8;
lam = 10; % [10,15,20] % [5,8,10] % 
dir_GT_info = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));

list_th_SNR = 4; % 3:5;
nSNR = length(list_th_SNR);
Table_all = zeros(1,nSNR*6);
for did = 1:nSNR
    th_SNR = list_th_SNR(did);
    dir_SUNS = fullfile(dir_parent, ['complete_TUnCaT_SF25\4816[1]th',num2str(th_SNR)]); % 4 v1
    dir_masks = fullfile(dir_SUNS, 'output_masks');
%     sub_folder = 'add_new_blockwise';
    sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
    dir_add_new = fullfile(dir_masks, sub_folder);
    sub0 = sprintf('trained dropout %gexp(-%g)',d0,lam);
    folder = fullfile(sub0,'accept all');
    if ~exist(fullfile(dir_add_new,folder),'dir')
        mkdir(fullfile(dir_add_new,folder));
    end
    dir_GT = fullfile(dir_parent, 'GT Masks'); % , sub_folder
%     dir_GT_info = fullfile(dir_video, 'GT info'); % , sub_folder
    % dir_GT = fullfile(dir_parent, 'GT Masks');
    % dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise'], 'GT Masks');

    load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
    mean([list_Recall,list_Precision,list_F1]);
    list_time_SUNS = list_time(:,end);
    [list_Recall, list_Precision, list_F1, list_Recall_add, list_Precision_add, ...
        list_Recall_add_max, list_Recall_max, list_Precision_max, list_F1_max, list_time_weights, ...
        list_time_classifier, list_time_merge] = deal(zeros(num_Exp,1));

    %% merge repeated neurons in list_added
    for eid = 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
        load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
            'masks_added_full','masks_added_crop','images_added_crop','time_weights'); % ,'list_valid'
        list_time_weights(eid) = time_weights;
        list_time_classifier(eid) = 0;

        tic;
        masks=permute(logical(Masks),[3,2,1]);
        list_added_all = masks_added_full;
        [Lx,Ly,num_added] = size(list_added_all);
        list_added_sparse = sparse(reshape(list_added_all,Lx*Ly,num_added));
        times = cell(1,num_added);
        [list_added_sparse_half,times] = piece_neurons_IOU(list_added_sparse,0.5,0.5,times);
        [list_added_sparse_final,times] = piece_neurons_consume(list_added_sparse_half,inf,0.5,0.75,times);
        list_added_final = reshape(full(list_added_sparse_final),Lx,Ly,[]);

        %%
    %     dir_masks = fullfile(dir_parent);
    %     load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
        Masks = cat(3,masks,list_added_final);
        list_time_merge(eid) = toc;
        save(fullfile(dir_add_new,folder,['Output_Masks_',Exp_ID,'_added.mat']),'Masks');
        Masks_2 = sparse(reshape(Masks,Lx*Ly,[]));
        n_init = size(masks,3);
        n_add = size(list_added_final,3);

        %%
        load(fullfile(dir_GT_info,['DroppedMasks_',Exp_ID,'.mat']),'DroppedMasks');
        masks_add_2 = sparse(reshape(DroppedMasks,Lx*Ly,[]));
%             load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'masks_add');
%             masks_add_2 = sparse(reshape(masks_add,Lx*Ly,[]));
        load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
%             load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_added_blockwise.mat']),'FinalMasks');
%             GTMasks_2 = sparse(reshape(FinalMasks,Lx*Ly,[]));
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
    disp(nanmean(Table(:,:),1));
    Table_all(1,(did-1)*6+(1:6)) = nanmean(Table_ext(end-1,1:6),1);
    save(fullfile(dir_add_new,folder,'eval.mat'),'list_Recall_add',...
        'list_Precision_add','list_Recall','list_Precision','list_F1','list_time');
%         disp('Finished this step');
end

