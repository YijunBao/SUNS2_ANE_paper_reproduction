% clear;
%% load data and GT masks
dir_parent='E:\data_CNMFE\';
addpath(genpath('E:\data_CNMFE'));
addpath(genpath('E:\data_CNMFE\Manual labeling GUI'));
addpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools');
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);
%%
eid = 1;
Exp_ID = list_Exp_ID{eid};
dir_masks = fullfile(dir_parent, 'GT Masks updated');
load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'_update_manual.mat']),'FinalMasks');

%% Remove empty or small regions
areas = squeeze(sum(sum(FinalMasks,1),2));
FinalMasks(:,:,areas<5)=[];
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

%% Merge highly overlapping neurons
ncells = size(FinalMasks,3);
times = cell(1,ncells);
FinalMasks2 = reshape(FinalMasks,Lx*Ly,ncells);
area2 = sum(FinalMasks2,1);
FinalMasks2(:,area2<5)=[];

% overlap = overlap_neurons_IOU(FinalMasks2,0.5,0.5,times)
overlap = overlap_neurons_consume(FinalMasks2,inf,0.5,0.75,times)

% [FinalMasks3,times] = piece_neurons_IOU(FinalMasks2,0.5,0.5,times);
% [FinalMasks4,times] = piece_neurons_consume(FinalMasks3,inf,0.5,0.75,times);
% [FinalMasks4,times] = piece_neurons_consume(FinalMasks2,inf,0.5,0.75,times);
% FinalMasks4=double(FinalMasks4>=0.5*max(FinalMasks4,[],1));

%%
load(fullfile(dir_parent,[Exp_ID,'.mat']),'Y','Ysiz');
%%
plot_masks_id(FinalMasks);
%%
DirSave = 'Results';
global result; result = [];
global resultString; resultString = {}; 
global resultSpikes; resultSpikes = {};
% main_SelectNeuron(Y,FinalMasks,DirSave,Exp_ID);
main_SelectNeuron(Y,FinalMasks(:,:,reshape(overlap',1,[])),DirSave,[Exp_ID,'_remove_overlap']);

% save(fullfile(dir_masks,['FinalMasks_',Exp_ID,'_merge.mat']),'FinalMasks');
% save(fullfile(data_path,'merged FinalMasks.mat'),'FinalMasks4');
% ncells = size(FinalMasks4,2);
