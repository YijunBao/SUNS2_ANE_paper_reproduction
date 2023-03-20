addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
% folder of the GT Masks
% dir_parent='E:\data_CNMFE\';
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);

% dir_parent='D:\data_TENASPIS\original_masks\';
dir_parent='D:\data_TENASPIS\added_refined_masks\';
d0 = 0.8;
lam = 20; % [10,15,20] % [5,8,10] % 
dir_GT_info = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));

% save_date = '20221221';
% dir_sub_save = ['cv_save_',save_date];
% dir_save = fullfile(dir_parent,'CNMFE');
% dir_eval='C:\Other methods\CNMF_E-1.1.2\';
save_date = '20230111';
dir_sub_save = ['cv_save_',save_date];
dir_save = fullfile(dir_parent,'min1pipe');
dir_eval='C:\Other methods\MIN1PIPE-3.0.0\';

optimize_th_cnn = false; % true; % 
list_th_SNR = 5; % 3:5;
nSNR = length(list_th_SNR);
res =  0; % [0,1,3:9]; %
num_frame = 0; % [0,2:8]; % 
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table_all = zeros(nv,nSNR*6);
Table_max = zeros(1,nSNR*3);
dir_masks = fullfile(dir_save,dir_sub_save); % 4 v1

for did = 1:nSNR
%     sub_folder = 'add_new_blockwise';
    sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
    dir_add_new = fullfile(dir_masks, sub_folder);
    dir_GT = fullfile(dir_parent, 'GT Masks'); % , sub_folder
%     dir_GT_info = fullfile(dir_video, 'GT info'); % , sub_folder
    % dir_GT = fullfile(dir_parent, 'GT Masks');
    % dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise'], 'GT Masks');

%     load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
%     mean([list_Recall,list_Precision,list_F1]);
    % list_time_SUNS = list_time(:,end);
    load(fullfile(dir_eval,['eval_TENASPIS_thb ',save_date,' cv 2round.mat']),'Table_time_ext');
    list_time_min1pipe = Table_time_ext(1:end-2,end-4);
    Table_time_ext(end-1,end-8:end-6);
    [list_Recall, list_Precision, list_F1, list_Recall_add, list_Precision_add, ...
        list_Recall_add_max, list_Recall_max, list_Precision_max, list_F1_max, list_time_weights, ...
        list_time_classifier, list_time_merge] = deal(zeros(num_Exp,1));

    %% merge repeated neurons in list_added
    for moid=1:length(list_mask_option)
        mask_option = list_mask_option{moid};
    for mcid=1:length(list_num_mask_channel)
        num_mask_channel = list_num_mask_channel(mcid);
        vid = (moid-1)*num_mc+(mcid-1);
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
        sub0 = sprintf('trained dropout %gexp(-%g)',d0,lam);
        folder = fullfile(sub0,sub1,sub2);
%         folder = ''; % 'classifier';
%         num_frame = version;
%         folder = ['avg_mask_0.5\classifier_res0_',num2str(version),'frames'];
%         folder = ['avg_mask_0.5\classifier_res',num2str(version),'_0+2 frames'];
        for eid = 1:num_Exp
            Exp_ID = list_Exp_ID{eid};
            load(fullfile(dir_masks,[Exp_ID,'_Masks.mat']),'Masks3');
%             load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
            load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
                'masks_added_full','masks_added_crop','images_added_crop','time_weights'); % ,'list_valid'
            list_time_weights(eid) = time_weights;
            load(fullfile(dir_add_new,folder,['CNN_predict_',Exp_ID,'_cv',num2str(eid-1),'.mat']), 'pred_valid','time_CNN'); 
            list_time_classifier(eid) = time_CNN;

            tic;
%             masks=permute(logical(Masks),[3,2,1]);
            masks=Masks3;
            list_added_all = masks_added_full(:,:,pred_valid);
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

            %% Calculate the recall without missing finder
            if vid==2
%                 dir_SUNS_parent = fullfile(dir_parent, ['complete_TUnCaT\4816[1]th',num2str(th_SNR)]); % 4 v1
%                 dir_masks_parent = fullfile(dir_SUNS_parent, 'output_masks');
%                 load(fullfile(dir_masks_parent,['Output_Masks_',Exp_ID(1:end-6),'.mat']),'Masks');
%                 Masks=permute(logical(Masks),[3,2,1]);
                Masks_2 = sparse(reshape(masks,Lx*Ly,[]));

                [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,0.5);
                list_Recall_max(eid) = Recall;
                list_Precision_max(eid) = Precision;
                list_F1_max(eid) = F1;
                [Recall_add_max, ~, ~, ~] = GetPerformance_Jaccard_2(masks_add_2,Masks_2,0.5);
                list_Recall_add_max(eid) = Recall_add_max;
            end
        end
        %%
        list_time = [list_time_min1pipe, list_time_weights, list_time_classifier, list_time_merge];
        list_time = [list_time, sum(list_time,2)];
        Table = [list_Recall_add,list_Precision_add,list_Recall,list_Precision,list_F1,list_time(:,end)];
        Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
        disp(nanmean(Table(:,:),1));
        Table_all(vid,(did-1)*6+(1:6)) = nanmean(Table_ext(end-1,1:6),1);
        save(fullfile(dir_add_new,folder,'eval.mat'),'list_Recall_add',...
            'list_Precision_add','list_Recall','list_Precision','list_F1','list_time');
%         disp('Finished this step');
        %%
        list_Precision_add_max = 0 * list_Recall_add_max;
        Table = [list_Recall_add_max,list_Precision_add_max,list_Recall_max,list_Precision_max,list_F1_max,list_time_min1pipe];
        Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];    
        Table_max(1,(did-1)*6+(1:6)) = nanmean(Table_ext(end-1,1:6),1);
    end
    end
end
Table_all_max = [Table_max; Table_all];