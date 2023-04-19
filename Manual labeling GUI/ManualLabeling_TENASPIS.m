clear
addpath(genpath('.'))
addpath(genpath(fullfile('..','ANE')))
global result; 
global resultString; 
global resultSpikes; 

%%
dir_parent=fullfile('..','data','data_TENASPIS','added_refined_masks');
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);

% name of file to read
% for eid = 1:num_Exp
eid = 1;
Exp_ID = list_Exp_ID{eid};
h5_name = fullfile(dir_parent,[Exp_ID,'.h5']);
Y = h5read(h5_name,'/mov');

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
