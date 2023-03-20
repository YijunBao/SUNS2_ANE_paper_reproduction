addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
scale_lowBG = 5e3;
scale_noise = 1;
results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_patch_dims = [253,316]; 
num_Exp = 10;

list_data_names={results_folder};
rate_hz = 10; % frame rate of each video
radius = 6;
data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\simulation_CNMFE_corr_noise',data_name);
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

%%
dir_parent=path_name;
d0 = 0.8;
list_lam = [5,8,10]; % 3,

optimize_th_cnn = true; % false; % 
list_th_SNR = 4; % 3:5;
nSNR = length(list_th_SNR);
res =  0; % [0,1,3:9]; %
num_frame = 0; % [0,2:8]; % 
mask_option = 'Xmask'; % {'nomask', 'mask', 'bmask', 'Xmask'}; % 
shuffle = ''; % '_shuffle'; % 
num_mask_channel = 1; % [1,2]; % 
nv = length(list_lam);
Table_all = zeros(nv,nSNR*3);
Table_max = zeros(1,nSNR*3);
for did = 1:nSNR
    th_SNR = list_th_SNR(did);
%     dir_SUNS = fullfile(dir_video, ['complete_TUnCaT\4816[1]th',num2str(th_SNR)]); % 4 v1
%     dir_masks = fullfile(dir_SUNS, 'output_masks');
%     dir_masks = fullfile(dir_parent, 'GT Masks');
%     sub_folder = 'add_new_blockwise';
    sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
%     dir_GT_info = fullfile(dir_video, 'GT info'); % , sub_folder
    % dir_GT = fullfile(dir_parent, 'GT Masks');
    % dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise'], 'GT Masks');

%     load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
%     mean([list_Recall,list_Precision,list_F1]);
%     list_time_SUNS = list_time(:,end);
    [list_Recall, list_Precision, list_F1, list_Recall_add, list_Recall_add_max, ...
        list_Recall_max, list_Precision_max, list_F1_max] = deal(zeros(num_Exp,1));

    %% merge repeated neurons in list_added
    for vid=1:length(list_lam)
        lam = list_lam(vid);
        dir_video = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
        dir_add_new = fullfile(dir_video, sub_folder);
        dir_GT = fullfile(dir_video, 'GT Masks'); % , sub_folder
        dir_masks = dir_video;
        if strcmp(mask_option,'nomask')
            if num_mask_channel == 1
                vid = vid+1;
            elseif num_mask_channel == 2
                continue;
            end
        end
        if num_frame <= 1
            img_option = 'avg';
            str_shuffle = '';
        else
            img_option = 'multi_frame';
            str_shuffle = shuffle;
        end
        sub1 = sprintf('%s_%s',img_option,mask_option);
        if optimize_th_cnn
            sub1 = [sub1,'_0.5'];
        end
        sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
        folder = [sub1,'\',sub2];
%         folder = ''; % 'classifier';
%         num_frame = version;
%         folder = ['avg_mask_0.5\classifier_res0_',num2str(version),'frames'];
%         folder = ['avg_mask_0.5\classifier_res',num2str(version),'_0+2 frames'];
        for eid = 1:num_Exp
            Exp_ID = list_Exp_ID{eid};
            load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
            load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
                'masks_added_full','masks_added_crop','images_added_crop'); % ,'list_valid'
            load(fullfile(dir_add_new,folder,['CNN_predict_',Exp_ID,'_cv',num2str(eid-1),'.mat']), 'pred_valid'); 

            tic;
%             masks=permute(logical(Masks),[3,2,1]);
            masks = FinalMasks;
            list_added_all = masks_added_full(:,:,pred_valid);
            [Lx,Ly,num_added] = size(list_added_all);
            list_added_sparse = sparse(reshape(list_added_all,Lx*Ly,num_added));
            times = cell(1,num_added);
            [list_added_sparse_half,times] = piece_neurons_IOU(list_added_sparse,0.5,0.5,times);
            [list_added_sparse_final,times] = piece_neurons_consume(list_added_sparse_half,inf,0.5,0.75,times);
            list_added_final = reshape(full(list_added_sparse_final),Lx,Ly,[]);

            %%
            Masks = cat(3,masks,list_added_final);
            save(fullfile(dir_add_new,folder,['Output_Masks_',Exp_ID,'_added.mat']),'Masks');
            list_added_final_2 = sparse(reshape(list_added_final,Lx*Ly,[]));

            %%
            load(fullfile(dir_masks,['DroppedMasks_',Exp_ID,'.mat']),'DroppedMasks');
            masks_add_2 = sparse(reshape(DroppedMasks,Lx*Ly,[]));
%             load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_added_sparse.mat']),'GTMasks_2');
%             load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_added_blockwise.mat']),'FinalMasks');
%             GTMasks_2 = sparse(reshape(FinalMasks,Lx*Ly,[]));
            [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add_2,list_added_final_2,0.5);
            list_Recall(eid) = Recall;
            list_Precision(eid) = Precision;
            list_F1(eid) = F1;
%             [Recall_add, ~, ~, ~] = GetPerformance_Jaccard_2(masks_add_2,Masks_2,0.5);
%             list_Recall_add(eid) = Recall_add;

            %% Calculate the recall when accepting all candidate missing neurons
            if vid==2
                [Lx,Ly,num_added] = size(masks_added_full);
                masks_added_sparse = sparse(reshape(masks_added_full,Lx*Ly,num_added));
                times = cell(1,num_added);
                [masks_added_sparse_half,times] = piece_neurons_IOU(masks_added_sparse,0.5,0.5,times);
                [masks_added_sparse_final,times] = piece_neurons_consume(masks_added_sparse_half,inf,0.5,0.75,times);
                masks_added_final = reshape(full(masks_added_sparse_final),Lx,Ly,[]);

                Masks_max = cat(3,masks,masks_added_final);
                save(fullfile(dir_add_new,folder,['Output_Masks_',Exp_ID,'_added_max.mat']),'Masks_max');
                masks_added_final_2 = sparse(reshape(masks_added_final,Lx*Ly,[]));

                [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add_2,masks_added_final_2,0.5);
                list_Recall_max(eid) = Recall;
                list_Precision_max(eid) = Precision;
                list_F1_max(eid) = F1;
%                 [Recall_add_max, ~, ~, ~] = GetPerformance_Jaccard_2(masks_add_2,Masks_max_2,0.5);
%                 list_Recall_add_max(eid) = Recall_add_max;
            end
        end
        %%
%         list_time = [list_time_SUNS, list_time_weights, list_time_classifier, list_time_merge];
%         list_time = [list_time, sum(list_time,2)];
        Table = [list_Recall,list_Precision,list_F1];
        Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
        disp(nanmean(Table(:,1:3),1));
        Table_all(vid,(did-1)*3+(1:3)) = nanmean(Table_ext(end-1,1:3),1);
        save(fullfile(dir_add_new,folder,'eval.mat'),...
            'list_Recall','list_Precision','list_F1'); % ,'list_Recall_add','list_time'
%         disp('Finished this step');
        %%
        Table = [list_Recall_max,list_Precision_max,list_F1_max];
        Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];    
        Table_max(1,(did-1)*3+(1:3)) = nanmean(Table_ext(end-1,1:3),1);
    end
end
Table_all_max = Table_all;
% Table_all_max = [Table_max; Table_all];
