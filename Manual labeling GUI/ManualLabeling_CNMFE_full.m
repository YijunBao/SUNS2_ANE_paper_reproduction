clear
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
if ~ exist(DirSave,'dir')
    mkdir(DirSave);
end

%%
% If there is an initial marking available, define it here. Otherwise,
% leave mask empty (mask = []).
FinalMasks = [];
result = [];
resultString = {}; 
resultSpikes = {};
% guitrace = [];

%% Load previous saved results
% load(fullfile(DirSave,['Added_',Exp_ID,'.mat']),'FinalMasks');
% load(fullfile(DirSave,['ManualIDs_traces_',Exp_ID,'.mat']),'result','resultString','resultSpikes'); %,'guitrace'
%%
main_SelectNeuron(Y,FinalMasks,DirSave,Exp_ID);
%%
% plot_masks_id(FinalMasks);
