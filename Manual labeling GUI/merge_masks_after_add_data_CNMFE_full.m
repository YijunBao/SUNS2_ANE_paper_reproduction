% clear;
addpath(genpath('.'))
addpath(genpath(fullfile('..','ANE')))
%%
dir_video=fullfile('E:','data_CNMFE');
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);
eid = 4;
Exp_ID = list_Exp_ID{eid};

DirSave = ['Results_',Exp_ID];
dir_add = fullfile(DirSave,'add_new_blockwise');
% if ~ exist(dir_add,'dir')
%     mkdir(dir_add);
% end

%%
load(fullfile(dir_add,[Exp_ID,'_added_auto_blockwise.mat']),'masks_added_full');
load(fullfile(dir_add, 'masks_processed.mat'), 'update_result');
list_valid = cellfun(@(x) (isempty(x) || x==1), update_result.list_valid);
FinalMasks = masks_added_full(:,:,list_valid);
save(fullfile(dir_add,[Exp_ID,'_confirmed.mat']),'FinalMasks');
% Y = h5read(fullfile(dir_video,[Exp_ID,'.h5']),'/mov');

%% Remove empty or small regions
[Lx,Ly,ncells] = size(FinalMasks);
for i = 1:ncells
    temp_edge = full(logical(FinalMasks(:,:,i)));
    CC = bwconncomp(temp_edge);
    if CC.NumObjects > 1
        areas = cellfun(@numel,CC.PixelIdxList);
        [~, ind] = max(areas);
        temp_correct = zeros(Lx, Ly);
        temp_correct(CC.PixelIdxList{ind}) = 1;
        FinalMasks(:,:,i) = temp_correct;
    end
end
areas = squeeze(sum(sum(FinalMasks,1),2));
FinalMasks(:,areas<5)=[];
FinalMasks2 = reshape(FinalMasks,Lx*Ly,ncells);

ncells = size(FinalMasks,3);
times = cell(1,ncells);

%% Merge highly overlapping neurons
[FinalMasks3,times] = piece_neurons_IOU(FinalMasks2,0.5,0.5,times);
[FinalMasks4,times] = piece_neurons_consume(FinalMasks3,inf,0.5,0.75,times);
% [FinalMasks4,times] = piece_neurons_consume(FinalMasks2,inf,0.5,0.75,times);
FinalMasks4=double(FinalMasks4>=0.5*max(FinalMasks4,[],1));
FinalMasks = reshape(FinalMasks4,Lx,Ly,[]);

save(fullfile(dir_add,[Exp_ID,'_added_merged.mat']),'FinalMasks');
ncells = size(FinalMasks4,2);

