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
dir_manual_draw = fullfile(DirSave,'manual_draw');
dir_merged_save = fullfile(DirSave,'manual_draw_remove_overlap');
if ~ exist(dir_merged_save,'dir')
    mkdir(dir_merged_save);
end

%%
load(fullfile(dir_manual_draw,['Added_',Exp_ID,'.mat']),'FinalMasks');
load(fullfile(dir_video, [Exp_ID,'.mat']),'Y','Ysiz'); %

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

% overlap = overlap_neurons_IOU(FinalMasks2,0.5,0.5,times)
overlap = overlap_neurons_consume(FinalMasks2,inf,0.5,0.75,times)
overlap_flat = reshape(overlap',1,[]);
Masks_nonoverlap = FinalMasks(:,:,setdiff(1:ncells,overlap_flat));
global result; result = [];
global resultString; resultString = {}; 
global resultSpikes; resultSpikes = {};
% main_SelectNeuron(Y,FinalMasks,DirSave,[Exp_ID,'_add3']);
main_SelectNeuron(Y,FinalMasks(:,:,overlap_flat),dir_merged_save,Exp_ID);

%% After finishing manual selection
load(fullfile(dir_merged_save,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
FinalMasks = cat(3,Masks_nonoverlap,FinalMasks);

%%
save(fullfile(dir_merged_save,['Manual_',Exp_ID,'_nonoverlap.mat']),'FinalMasks');
