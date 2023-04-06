clear
dir_video='E:\data_CNMFE\';
addpath(genpath('E:\data_CNMFE\Manual labeling GUI'));
addpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools');
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);

% name of file to read
eid = 4;
Exp_ID = list_Exp_ID{eid};
load([dir_video,Exp_ID,'.mat'],'Y','Ysiz'); %

DirSave = 'Results';
if ~ exist(DirSave,'dir')
    mkdir(DirSave);
end
% If there is an initial marking available, define it here. Otherwise,
% leave mask empty (mask = []).
FinalMasks = [];
global result; result = [];
global resultString; resultString = {}; 
global resultSpikes; resultSpikes = {};
% guitrace = [];

%%
load([dir_video,DirSave,'\Added_',Exp_ID,'.mat'],'FinalMasks');
load([dir_video,DirSave,'\ManualIDs_traces_',Exp_ID,'.mat'],'result','resultString','resultSpikes'); %,'guitrace'

main_SelectNeuron(Y,FinalMasks,DirSave,Exp_ID);

%%
% plot_masks_id(FinalMasks);
