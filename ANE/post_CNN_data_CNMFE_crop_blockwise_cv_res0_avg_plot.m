addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% green = [0.1,0.9,0.1]; % color(5,:); %
% red = [0.9,0.1,0.1]; % color(7,:); %
% blue = [0.1,0.8,0.9]; % color(6,:); %
yellow = [0.8,0.8,0.0]; % color(3,:); %
magenta = [0.9,0.3,0.9]; % color(4,:); %
green = [0.0,0.65,0.0]; % color(5,:); %
red = [0.8,0.0,0.0]; % color(7,:); %
blue = [0.0,0.6,0.8]; % color(6,:); %
colors_multi = distinguishable_colors(16);

%%
% folder of the GT Masks
% dir_parent='E:\data_CNMFE\';
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_title = {'upper left', 'upper right', 'lower left', 'lower right'};

data_ind = 2;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
save_figures = 0;

dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise_weighted_sum_unmask']);
dir_video_SNR = fullfile(dir_parent,'complete_TUnCaT\network_input\');
SNR_range = [2,10];

optimize_th_cnn = false; % true; % 
list_th_SNR = 4; % 3:5;
nSNR = length(list_th_SNR);
res =  0; % [0,1,3:9]; %
num_frame = 0; % [0,2:8]; % 
list_mask_option = {'bmask'}; % {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = 1; % [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table_all = zeros(nv,nSNR*4);
Table_max = zeros(1,nSNR*3);
for did = 1:nSNR
    th_SNR = list_th_SNR(did);
    dir_SUNS = fullfile(dir_parent, ['complete_TUnCaT\4816[1]th',num2str(th_SNR)]); % 4 v1
    dir_masks = fullfile(dir_SUNS, 'output_masks');
%     sub_folder = 'add_new_blockwise';
    sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
    dir_add_new = fullfile(dir_masks, sub_folder);
    dir_GT = fullfile(dir_parent,'GT Masks', sub_folder);
    dir_GT_original = fullfile(dir_parent, 'GT Masks');
    % dir_GT = fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise'], 'GT Masks');

    load(fullfile(dir_masks,'Output_Info_All.mat'),'list_time','list_Recall','list_Precision','list_F1');
    mean([list_Recall,list_Precision,list_F1]);
    list_time_SUNS = list_time(:,end);
    [list_Recall, list_Precision, list_F1, list_Recall_add, list_Recall_add_max, ...
        list_Recall_max, list_Precision_max, list_F1_max, list_time_weights, ...
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
        folder = [sub1,'\',sub2];
%         folder = ''; % 'classifier';
%         num_frame = version;
%         folder = ['avg_mask_0.5\classifier_res0_',num2str(version),'frames'];
%         folder = ['avg_mask_0.5\classifier_res',num2str(version),'_0+2 frames'];
        for eid = 1:num_Exp
            Exp_ID = list_Exp_ID{eid};
            video_SNR = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input'); % raw_traces
            SNR_max = max(video_SNR,[],3);
            load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
            load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
                'masks_added_full','masks_added_crop','images_added_crop','time_weights'); % ,'list_valid'
            list_time_weights(eid) = time_weights;
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
        %     dir_masks = fullfile(dir_parent);
        %     load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
            Masks = cat(3,masks,list_added_final);
            list_time_merge(eid) = toc;
            % save(fullfile(dir_add_new,folder,['Output_Masks_',Exp_ID,'_added.mat']),'Masks');
            Masks_2 = sparse(reshape(Masks,Lx*Ly,[]));

            %%
%             load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_sparse.mat']),'GTMasks_2');
            load(fullfile(dir_GT_original,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
            GTMasks_2_original = sparse(reshape(FinalMasks,Lx*Ly,[]));
            load(fullfile(dir_GT,['FinalMasks_',Exp_ID,'_added_blockwise.mat']),'FinalMasks');
            GTMasks_2 = sparse(reshape(FinalMasks,Lx*Ly,[]));
            masks_add_2 = GTMasks_2(:,size(GTMasks_2_original,2)+1:end);
            [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,Masks_2,0.5);
            list_Recall(eid) = Recall;
            list_Precision(eid) = Precision;
            list_F1(eid) = F1;
            [Recall_add, ~, ~, ~] = GetPerformance_Jaccard_2(masks_add_2,Masks_2,0.5);
            list_Recall_add(eid) = Recall_add;


            %% Plot summary images and masks
            masks_original = reshape(full(GTMasks_2_original),Lx,Ly,[]);
            edge_masks_original = 0*masks_original;
            for nn = 1:size(masks_original,3)
                edge_masks_original(:,:,nn) = edge(masks_original(:,:,nn));
            end
            masks_original_sum = sum(masks_original,3);
            edge_masks_original_sum = sum(edge_masks_original,3);

            masks_SUNS = masks;
            edge_masks_SUNS = 0*masks_SUNS;
            for nn = 1:size(masks_SUNS,3)
                edge_masks_SUNS(:,:,nn) = edge(masks_SUNS(:,:,nn));
            end
            masks_SUNS_sum = sum(masks_SUNS,3);
            edge_masks_SUNS_sum = sum(edge_masks_SUNS,3);

            masks_add_GT = reshape(full(masks_add_2),Lx,Ly,[]);
            edge_masks_add_GT = 0*masks_add_GT;
            for nn = 1:size(masks_add_GT,3)
                edge_masks_add_GT(:,:,nn) = edge(masks_add_GT(:,:,nn));
            end
            masks_add_GT_sum = sum(masks_add_GT,3);
            edge_masks_add_GT_sum = sum(edge_masks_add_GT,3);

            masks_add_found = list_added_final;
            edge_masks_add_found = 0*masks_add_found;
            for nn = 1:size(masks_add_found,3)
                edge_masks_add_found(:,:,nn) = edge(masks_add_found(:,:,nn));
            end
            masks_add_found_sum = sum(masks_add_found,3);
            edge_masks_add_found_sum = sum(edge_masks_add_found,3);

            mag = 1;
            figure('Position',[50,50,400,300],'Color','w');
            %     imshow(raw_max,[0,1024]);
            imagesc(SNR_max,SNR_range); 
            axis('image'); colormap gray;
            xticklabels({}); yticklabels({});
            hold on;
            alphaImg_original = ones(Lx*mag,Ly*mag).*reshape(yellow,1,1,3);
            alphaImg_add_GT = ones(Lx*mag,Ly*mag).*reshape(blue,1,1,3);
            alphaImg_SUNS= ones(Lx*mag,Ly*mag).*reshape(green,1,1,3);
            alphaImg_add_found = ones(Lx*mag,Ly*mag).*reshape(red,1,1,3);
            alpha = 0.8;
            image(alphaImg_original,'Alphadata',alpha*(edge_masks_original_sum));  
            image(alphaImg_add_GT,'Alphadata',alpha*(edge_masks_add_GT_sum));  
            image(alphaImg_SUNS,'Alphadata',alpha*(edge_masks_SUNS_sum));  
            image(alphaImg_add_found,'Alphadata',alpha*(edge_masks_add_found_sum));  
            title(list_title{eid});
            h=colorbar;
            set(get(h,'Label'),'String','Peak SNR');
            set(h,'FontSize',12);
            if save_figures
                saveas(gcf,[Exp_ID, ' SUNS + missing_finder.png']);
                % image(alphaImg_GT,'Alphadata',alpha*(edge_masks_GT_sum));  
                % saveas(gcf,[Exp_ID, 'missing_finder .png']);
                % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
            end
        end
        %%
        % list_time = [list_time_SUNS, list_time_weights, list_time_classifier, list_time_merge];
        % list_time = [list_time, sum(list_time,2)];
        Table = [list_Recall_add,list_Recall,list_Precision,list_F1,list_time(:,[1,end])];
        Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
%         disp(nanmean(Table(:,1:4),1));
%         Table_all(vid,(did-1)*4+(1:4)) = nanmean(Table_ext(end-1,1:4),1);
%         save(fullfile(dir_add_new,folder,'eval.mat'),'list_Recall_add',...
%             'list_Recall','list_Precision','list_F1','list_time');
% %         disp('Finished this step');
%         %%
%         Table = [list_Recall_add_max,list_Recall_max,list_Precision_max,list_F1_max];
%         Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];    
%         Table_max(1,(did-1)*4+(1:4)) = nanmean(Table_ext(end-1,1:4),1);
    end
    end
end
% Table_all_max = [Table_max; Table_all];