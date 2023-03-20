%%
% folder of the GT Masks
% dir_parent='E:\data_CNMFE\';
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};

data_ind = 2;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);

dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_added_auto']);

list_th_SNR = 3:5;
nSNR = length(list_th_SNR);
list_res = 0; % [0,1,3:9]; % 
list_nframes = [0,2:8]; % 0; % 
num_res = length(list_res);
num_nf = length(list_nframes);
nv = max(num_res,num_nf);
Table_all = zeros(nv,nSNR*3);
mask_option = 'mask'; % 'nomask', 'mask', 'bmask', 'Xmask';
shuffle = ''; % '_shuffle'; % 
num_mask_channel = 2; % 1; % 
optimize_th_cnn = false; % true; % 
for did = 1:nSNR
    th_SNR = list_th_SNR(did);
    dir_SUNS = fullfile(dir_parent, ['complete_TUnCaT\4816[1]th',num2str(th_SNR)]); % 4 v1
    dir_masks = fullfile(dir_SUNS, 'output_masks');
    dir_add_new = fullfile(dir_masks, 'add_new_blockwise');
%     dir_GT = fullfile(dir_parent, 'GT Masks');
    dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise'], 'GT Masks');

    load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
    mean([list_Recall,list_Precision,list_F1]);
    list_time_SUNS = list_time(:,end);
    [list_Recall, list_Precision, list_F1, list_time_weights, ...
        list_time_classifier, list_time_merge] = deal(zeros(num_Exp,1));

    %% merge repeated neurons in list_added
    for rid=1:num_res
        res = list_res(rid);
    for nid=1:num_nf
        num_frame = list_nframes(nid);
        vid = max(rid,nid);
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
            load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
            load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
                'masks_added_full','masks_added_crop','images_added_crop','time_weights'); % ,'list_valid'
            list_time_weights(eid) = time_weights;
            load(fullfile(dir_add_new,folder,['CNN_predict_',Exp_ID,'.mat']), 'pred_valid','time_CNN'); 
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
        %     dir_masks = fullfile(dir_parent);
        %     load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
            Masks = cat(3,masks,list_added_final);
            list_time_merge(eid) = toc;
            save(fullfile(dir_add_new,folder,['Output_Masks_',Exp_ID,'_added.mat']),'Masks');
            Masks_2 = sparse(reshape(Masks,Lx*Ly,[]));

            %%
            load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
            [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,0.5);
            list_Recall(eid) = Recall;
            list_Precision(eid) = Precision;
            list_F1(eid) = F1;
        end
        %%
        list_time = [list_time_SUNS, list_time_weights, list_time_classifier, list_time_merge];
        list_time = [list_time, sum(list_time,2)];
        Table = [list_Recall,list_Precision,list_F1,list_time(:,[1,end])];
        Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
        disp(nanmean(Table(:,1:3),1));
        Table_all(vid,(did-1)*3+(1:3)) = nanmean(Table_ext(end-1,1:3),1);
        save(fullfile(dir_add_new,folder,'eval.mat'),'list_Recall','list_Precision','list_F1','list_time');
%         disp('Finished this step');
    end
    end
end