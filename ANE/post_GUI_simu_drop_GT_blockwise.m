addpath(genpath('.'))

%%
scale_lowBG = 1e3;
scale_noise = 0.3;
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
avg_radius = radius; % original_masks
% avg_radius = 9; % added_refined_masks
r_bg_ratio = 3;
% r_bg_ext = list_avg_radius(data_ind) * (r_bg_ratio+1);

%%
d0 = 0.8;
for lam = [3,5,8,10] % 
for vid=1:num_Exp
    Exp_ID = list_Exp_ID{vid};
    dir_parent=path_name;
    dir_masks = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
    dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');

    %%
    load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations')
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    masks = FinalMasks;
    load(fullfile(dir_masks,['DroppedMasks_',Exp_ID,'.mat']),'DroppedMasks');
    masks_add_GT = DroppedMasks;
%     load(fullfile(dir_add_new,[Exp_ID,'_processed.mat']),'update_result')
%     list_processed = update_result.list_processed;
%     ListStrings = update_result.ListStrings;
%     list_valid = cell2mat(update_result.list_valid);
%     list_select_frames = update_result.list_select_frames;
%     masks_added_crop_update = cat(3,update_result.list_mask_update{:});

    %% Find valid neurons
    [Lx, Ly, ~] = size(masks);
    mask_new_full_2 = reshape(masks_added_full,Lx*Ly,[]);
    mask_add_full_2 = reshape(masks_add_GT,Lx*Ly,[]);
%     [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(mask_add_full_2,mask_new_full_2,0.1);
    IoU = 1 - JaccardDist_2(mask_add_full_2,mask_new_full_2);
    list_valid = any(IoU > 0.3,1);
%     list_list_valid{nn} = list_valid;

    %%
    figure; imagesc(sum(masks_added_full,3)); axis image; colorbar;
    figure; imagesc(sum(masks_add_GT,3)); axis image; colorbar;

    %% Calculate the neighboring neurons, and updating masks
    N = length(added_frames);
    list_neighbors = cell(1,N);
    masks_sum = sum(masks,3);
%     masks_added_full_update = masks_added_full;
    
    for n = 1:N
        loc = patch_locations(n,:);
        list_neighbors{n} = masks_sum(loc(1):loc(2),loc(3):loc(4));
%         masks_added_full_update(loc(1):loc(2),loc(3):loc(4),n)=masks_added_crop_update(:,:,n);
    end
    masks_neighbors_crop = cat(3,list_neighbors{:});
    
    %%
    save(fullfile(dir_add_new,[Exp_ID,'_added_CNNtrain_blockwise.mat']), ...
        'added_frames','added_weights','list_valid','masks_neighbors_crop',...
        'masks_added_crop','images_added_crop'); % 

    %% merge repeated neurons in list_added
%     masks_added_full_valid = masks_added_full_update(:,:,list_valid);
%     [Lx,Ly,num_added] = size(masks_added_full_valid);
%     list_added_sparse = sparse(reshape(masks_added_full_valid,Lx*Ly,num_added));
%     times = cell(1,num_added);
%     [list_added_sparse_half,times] = piece_neurons_IOU(list_added_sparse,0.5,0.5,times);
%     [list_added_sparse_final,times] = piece_neurons_consume(list_added_sparse_half,inf,0.5,0.75,times);
%     list_added_final = reshape(full(list_added_sparse_final),Lx,Ly,[]);
% 
%     figure; imagesc(sum(masks_added_full_valid,3)); axis image; colorbar;
%     fprintf('%d->%d\n',num_added,size(list_added_final,3));
% 
%     %%
%     dir_masks = fullfile(dir_parent,'GT Masks');
%     load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
%     FinalMasks = cat(3,FinalMasks,list_added_final);
%     save(fullfile(dir_add_new,['FinalMasks_',Exp_ID,'_added_blockwise.mat']),'FinalMasks');
end
end