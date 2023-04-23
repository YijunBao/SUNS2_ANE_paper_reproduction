clear
addpath(genpath('.'))
addpath(genpath(fullfile('..','ANE')))
global result; 
global resultString; 
global resultSpikes; 

%%
dir_video=fullfile('E:','data_CNMFE');
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);

% name of file to read
% for eid = 1:num_Exp
eid = 4;
Exp_ID = list_Exp_ID{eid};
load(fullfile(dir_video, [Exp_ID,'.mat']),'Y','Ysiz'); %

DirSave = ['Results_',Exp_ID];
dir_manual_draw = fullfile(DirSave,'manual_draw');
if ~ exist(dir_manual_draw,'dir')
    mkdir(dir_manual_draw);
end

%%
% If there is an initial marking available, define it here. Otherwise,
% leave mask empty (mask = []).
FinalMasks = [];
result = [];
resultString = {}; 
resultSpikes = {};
% guitrace = [];

%% If using a previously saved result, load it first.
% load(fullfile(DirSave,['Added_',Exp_ID,'.mat']),'FinalMasks');
% load(fullfile(DirSave,['ManualIDs_traces_',Exp_ID,'.mat']),'result','resultString','resultSpikes'); %,'guitrace'
%%
main_SelectNeuron(Y,FinalMasks,dir_manual_draw,Exp_ID);
%%
% plot_masks_id(FinalMasks);
