clear
dir_video='D:\data_TENASPIS\';
addpath(genpath('E:\data_CNMFE\Manual labeling GUI'));
addpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools');
list_Exp_ID={'Mouse_1 recording_20141122_083913-002_normcorr',...
    'Mouse_2 recording_20141218_095135-002_normcorr',...
    'Mouse_3 recording_20150831_083050-002_normcorr',...
    'Mouse_4 recording_20150901_103340-002_normcorr',...
    'Mouse_1M recording_20151202_172313-006_normcorr',...
    'Mouse_2M recording_20160421_125208-007_normcorr',...
    'Mouse_3M recording_20160706_171514-006_normcorr',...
    'Mouse_4M recording_20160708_172756-007_normcorr'};
num_Exp = length(list_Exp_ID);

% name of file to read
eid = 8;
Exp_ID = list_Exp_ID{eid};
load([dir_video,Exp_ID,'.mat'],'Mpr'); %

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

% Y = permute(Mpr(:,:,201:end),[2,1,3]);
Y = permute(Mpr,[2,1,3]);

%%
load([dir_video,DirSave,'\Added_',Exp_ID,'.mat'],'FinalMasks');
load([dir_video,DirSave,'\ManualIDs_traces_',Exp_ID,'.mat'],'result','resultString','resultSpikes'); %,'guitrace'

main_SelectNeuron(Y,FinalMasks,DirSave,Exp_ID);

%%
% plot_masks_id(FinalMasks);

%% run after crop_video
ind=7;
data_name = list_Exp_ID{ind};
h5_name = fullfile(dir_parent,sub,[data_name,'.h5']);
Y = h5read(h5_name,'/mov');
DirSave = 'Results';
Exp_ID = list_Exp_ID{ind};
dir_video='D:\data_TENASPIS\';
%%
FinalMasks = [];
global result; result = [];
global resultString; resultString = {}; 
global resultSpikes; resultSpikes = {};
DirSave = 'Results';
Exp_ID = list_Exp_ID{ind};
dir_video='D:\data_TENASPIS\';
Y = mov;
main_SelectNeuron(Y,FinalMasks,DirSave,Exp_ID);
