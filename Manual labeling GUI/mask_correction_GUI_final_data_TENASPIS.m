% load('.\Result_PFC4_15Hz\masks_update(1--237)-1+3.mat','update_result')
%%
% folder of the GT Masks
dir_parent='D:\data_TENASPIS\original_masks\';
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};

vid=5;
Exp_ID = list_Exp_ID{vid};

masks_update = update_result.masks_update;
list_delete = update_result.list_delete;
list_added = update_result.list_added;
ListStrings = update_result.ListStrings;
list_IoU = update_result.list_IoU;
list_avg_frame = update_result.list_avg_frame;
list_mask_update = update_result.list_mask_update;

%% merge repeated neurons in list_added
list_added_all = cell2mat(reshape(list_added,1,1,[]));
[Lx,Ly,num_added] = size(list_added_all);
list_added_sparse = sparse(reshape(list_added_all,Lx*Ly,num_added));
times = cell(1,num_added);
[list_added_sparse_half,times] = piece_neurons_IOU(list_added_sparse,0.5,0.5,times);
[list_added_sparse_final,times] = piece_neurons_consume(list_added_sparse_half,inf,0.5,0.75,times);
list_added_final = reshape(full(list_added_sparse_final),Lx,Ly,[]);

%%
dir_save = fullfile(dir_parent,'GT Masks refined');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end
FinalMasks = cat(3,masks_update(:,:,~list_delete),list_added_final);
save(fullfile(dir_save,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
