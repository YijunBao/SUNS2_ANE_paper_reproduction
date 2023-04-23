addpath(genpath('.'))
addpath(genpath(fullfile('..','ANE')))
%%
% folder of the GT Masks
dir_parent=fullfile('..','data','data_TENASPIS','added_refined_masks');
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};

vid=1;
Exp_ID = list_Exp_ID{vid};
DirSave = ['Results_',Exp_ID];
dir_refine = fullfile(DirSave,'refined');
load(fullfile(dir_refine,'masks_update.mat'),'update_result');

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
FinalMasks = masks_update(:,:,~list_delete);
if ~isempty(list_added_final)
    FinalMasks = cat(3,FinalMasks,list_added_final);
end
save(fullfile(DirSave,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
