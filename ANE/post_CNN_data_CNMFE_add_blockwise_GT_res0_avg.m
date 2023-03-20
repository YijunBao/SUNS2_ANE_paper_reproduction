addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
% folder of the GT Masks
% dir_parent='E:\data_CNMFE\';
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};

data_ind = 2;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false); % ,'_added'
num_Exp = length(list_Exp_ID);

dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise_weighted_sum_unmask']);
neuron_amp = 0.001; % [0.001, 0.002, 0.003, 0.005];
dir_video = fullfile(dir_parent, ['add_neurons_',num2str(neuron_amp),'_rotate']);

optimize_th_cnn = false; % true; % 
list_th_SNR = 4; % 3:5;
nSNR = length(list_th_SNR);
res =  0; % [0,1,3:9]; %
num_frame = 0; % [0,2:8]; % 
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table_all = zeros(nv,nSNR*3);
Table_max = zeros(1,nSNR*3);
for did = 1:nSNR
    th_SNR = list_th_SNR(did);
%     dir_SUNS = fullfile(dir_video, ['complete_TUnCaT\4816[1]th',num2str(th_SNR)]); % 4 v1
%     dir_masks = fullfile(dir_SUNS, 'output_masks');
    dir_masks = fullfile(dir_parent, 'GT Masks');
%     sub_folder = 'add_new_blockwise';
    sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
    dir_add_new = fullfile(dir_video, sub_folder);
    dir_GT = fullfile(dir_video, 'GT Masks'); % , sub_folder
    dir_GT_info = fullfile(dir_video, 'GT info'); % , sub_folder
    % dir_GT = fullfile(dir_parent, 'GT Masks');
    % dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise'], 'GT Masks');

%     load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
%     mean([list_Recall,list_Precision,list_F1]);
%     list_time_SUNS = list_time(:,end);
    [list_Recall, list_Precision, list_F1, list_Recall_add, list_Recall_add_max, ...
        list_Recall_max, list_Precision_max, list_F1_max] = deal(zeros(num_Exp,1));

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
            load(fullfile(dir_GT_info,['GT_',Exp_ID,'_added.mat']),'masks_add');
            masks_add_2 = sparse(reshape(masks_add,Lx*Ly,[]));
%             load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_added_sparse.mat']),'GTMasks_2');
%             load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_added_blockwise.mat']),'FinalMasks');
%             GTMasks_2 = sparse(reshape(FinalMasks,Lx*Ly,[]));
            [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(masks_add_2,list_added_final_2,0.5);
            list_Recall(eid) = Recall;
            list_Precision(eid) = Precision;
            list_F1(eid) = F1;
%             disp([size(m),sum(m,'all')])
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
%                 disp([size(m),sum(m,'all')])
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
end
Table_all_max = [Table_max; Table_all];
