% data_ind = 4;
addpath(genpath('.'))

%%
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_avg_radius = [5,6,8,14];
list_lam = [15,5,8,8];
r_bg_ratio = 3;

data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
d0 = 0.8;
sub_added = '';
lam = list_lam(data_ind);

%%
for vid=1:4
    Exp_ID = list_Exp_ID{vid};
    dir_parent = fullfile('..','data','data_CNMFE',[data_name,sub_added]);
    dir_masks = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
    dir_add_new = fullfile(dir_masks, 'add_new_blockwise');

    %%
    load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations')
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    masks = FinalMasks;
    load(fullfile(dir_masks,['DroppedMasks_',Exp_ID,'.mat']),'DroppedMasks');
    masks_add_GT = DroppedMasks;

    %% Find valid neurons
    [Lx, Ly, ~] = size(masks);
    mask_new_full_2 = reshape(masks_added_full,Lx*Ly,[]);
    mask_add_full_2 = reshape(masks_add_GT,Lx*Ly,[]);
%     [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(mask_add_full_2,mask_new_full_2,0.1);
    IoU = 1 - JaccardDist_2(mask_add_full_2,mask_new_full_2);
    list_valid = any(IoU > 0.3,1);

    %%
%     figure; imagesc(sum(masks_added_full,3)); axis image; colorbar;
%     figure; imagesc(sum(masks_add_GT,3)); axis image; colorbar;

    %% Calculate the neighboring neurons, and updating masks
    N = length(added_frames);
    list_neighbors = cell(1,N);
    masks_sum = sum(masks,3);
    
    for n = 1:N
        loc = patch_locations(n,:);
        list_neighbors{n} = masks_sum(loc(1):loc(2),loc(3):loc(4));
    end
    masks_neighbors_crop = cat(3,list_neighbors{:});
    
    %%
    save(fullfile(dir_add_new,[Exp_ID,'_added_CNNtrain_blockwise.mat']), ...
        'added_frames','added_weights','list_valid','masks_neighbors_crop',...
        'masks_added_crop','images_added_crop'); % 
end
