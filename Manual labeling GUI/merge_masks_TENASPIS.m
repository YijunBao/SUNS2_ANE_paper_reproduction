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

% dir_masks_old = fullfile(dir_video,'GT Masks include_large');
dir_merged_save = [DirSave,'_remove_overlap'];
if ~ exist(dir_merged_save,'dir')
    mkdir(dir_merged_save);
end

%%
% load(fullfile(DirSave,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
load(fullfile(DirSave,['Added_',Exp_ID,'.mat']),'FinalMasks');
Y = h5read(fullfile(dir_video,[Exp_ID,'.h5']),'/mov');

%% Remove empty or small regions
areas = squeeze(sum(sum(FinalMasks,1),2));
FinalMasks(:,areas<5)=[];
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

%% Automatic merging
% [FinalMasks3,times] = piece_neurons_IOU(FinalMasks2,0.5,0.5,times);
% [FinalMasks4,times] = piece_neurons_consume(FinalMasks3,inf,0.5,0.75,times);
% [FinalMasks4,times] = piece_neurons_consume(FinalMasks2,inf,0.5,0.75,times);
% FinalMasks4=double(FinalMasks4>=0.5*max(FinalMasks4,[],1));

% save(fullfile(data_path,'merged FinalMasks.mat'),'FinalMasks4');
% ncells = size(FinalMasks4,2);
%% Manual merging
% overlap = overlap_neurons_IOU(FinalMasks2,0.5,0.5,times)
overlap = overlap_neurons_consume(FinalMasks2,inf,0.5,0.75,times)
overlap_flat = reshape(overlap',1,[]);
global result; result = [];
global resultString; resultString = {}; 
global resultSpikes; resultSpikes = {};
% main_SelectNeuron(Y,FinalMasks,DirSave,[Exp_ID,'_add3']);
main_SelectNeuron(Y,FinalMasks(:,:,overlap_flat),dir_merged_save,Exp_ID);

%% After finishing manual selection
load(fullfile(dir_merged_save,['Added_',Exp_ID,'.mat']),'FinalMasks');
