% clear;
addpath(genpath('.'))
addpath(genpath(fullfile('..','ANE')))
%%
dir_video=fullfile('..','data','data_TENASPIS','added_refined_masks');
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
eid = 1;
Exp_ID = list_Exp_ID{eid};

DirSave = ['Results_',Exp_ID];
dir_add = fullfile(DirSave,'add_new_blockwise');
dir_manual_merged = fullfile(DirSave,'manual_draw_remove_overlap');
dir_refine = fullfile(DirSave,'refined');
if ~ exist(dir_refine,'dir')
    mkdir(dir_refine);
end

%% 
load(fullfile(dir_manual_merged,['Manual_',Exp_ID,'_nonoverlap.mat']),'FinalMasks');
FinalMasks_manual = FinalMasks;
load(fullfile(dir_add,[Exp_ID,'_added_merged.mat']),'FinalMasks');
FinalMasks_added = FinalMasks;
FinalMasks = cat(3,FinalMasks_manual,FinalMasks_added);

%%
save(fullfile(dir_refine,[Exp_ID,'_manual_added.mat']),'FinalMasks');

