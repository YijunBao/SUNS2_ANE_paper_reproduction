%% TENASPIS dataset
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
DirData = 'D:\data_TENASPIS\added_refined_masks';

dir_parent = fullfile(DirData, 'complete_TUnCaT_SF25');
dir_network_input = fullfile(dir_parent,'network_input');

dir_raw_max = 'TENASPIS mat\raw_max';
if ~exist(dir_raw_max,'dir')
    mkdir(dir_raw_max)
end
dir_SNR_max = 'TENASPIS mat\SNR_max';
if ~exist(dir_SNR_max,'dir')
    mkdir(dir_SNR_max)
end

for k=1:num_Exp
    clear video_SNR video_raw
    Exp_ID = list_Exp_ID{k};
    
    video_raw=h5read(fullfile(DirData,[Exp_ID,'.h5']), '/mov');
    raw_max = max(video_raw,[],3);
    save(fullfile(dir_raw_max,['raw_max_',Exp_ID,'.mat']),'raw_max');
%     load(['mat\raw_max_',Exp_ID,'.mat'],'raw_max');

    video_SNR=h5read(fullfile(dir_network_input,[Exp_ID,'.h5']), '/network_input');
    SNR_max = max(video_SNR,[],3);
    save(fullfile(dir_SNR_max,['SNR_max_',Exp_ID,'.mat']),'SNR_max');
%     load(['mat\SNR_max_',Exp_ID,'.mat'],'SNR_max');
end

%% Simulated dataset
scale_lowBG = 5e3;
scale_noise = 1;
results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_data_names={results_folder};
data_ind = 1;
data_name = list_data_names{data_ind};
DirData = fullfile('E:\simulation_CNMFE_corr_noise',data_name);

num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);

dir_parent = fullfile(DirData, 'complete_TUnCaT');
dir_network_input = fullfile(dir_parent,'network_input');

dir_raw_max = [results_folder,' mat\raw_max'];
if ~exist(dir_raw_max,'dir')
    mkdir(dir_raw_max)
end
dir_SNR_max = [results_folder,' mat\SNR_max'];
if ~exist(dir_SNR_max,'dir')
    mkdir(dir_SNR_max)
end

for k=1:num_Exp
    clear video_SNR video_raw
    Exp_ID = list_Exp_ID{k};
    
    video_raw=h5read(fullfile(DirData,[Exp_ID,'.h5']), '/mov');
    raw_max = max(video_raw,[],3);
    save(fullfile(dir_raw_max,['raw_max_',Exp_ID,'.mat']),'raw_max');
%     load(['mat\raw_max_',Exp_ID,'.mat'],'raw_max');

    video_SNR=h5read(fullfile(dir_network_input,[Exp_ID,'.h5']), '/network_input');
    SNR_max = max(video_SNR,[],3);
    save(fullfile(dir_SNR_max,['SNR_max_',Exp_ID,'.mat']),'SNR_max');
%     load(['mat\SNR_max_',Exp_ID,'.mat'],'SNR_max');
end

%% CNMF-E dataset
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 4;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
DirData = fullfile('E:\data_CNMFE',data_name);

dir_parent = fullfile(DirData, 'complete_TUnCaT_SF50');
dir_network_input = fullfile(dir_parent,'network_input');

dir_raw_max = 'CNMFE mat\raw_max';
if ~exist(dir_raw_max,'dir')
    mkdir(dir_raw_max)
end
dir_SNR_max = 'CNMFE mat\SNR_max';
if ~exist(dir_SNR_max,'dir')
    mkdir(dir_SNR_max)
end

for k=1:num_Exp
    clear video_SNR video_raw
    Exp_ID = list_Exp_ID{k};
    
    video_raw=h5read(fullfile(DirData,[Exp_ID,'.h5']), '/mov');
    raw_max = max(video_raw,[],3);
    save(fullfile(dir_raw_max,['raw_max_',Exp_ID,'.mat']),'raw_max');
%     load(['mat\raw_max_',Exp_ID,'.mat'],'raw_max');

    video_SNR=h5read(fullfile(dir_network_input,[Exp_ID,'.h5']), '/network_input');
    SNR_max = max(video_SNR,[],3);
    save(fullfile(dir_SNR_max,['SNR_max_',Exp_ID,'.mat']),'SNR_max');
%     load(['mat\SNR_max_',Exp_ID,'.mat'],'SNR_max');
end

%% CNMF-E dataset full videos
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);
DirData = 'E:\data_CNMFE\full videos';

dir_parent = fullfile(DirData, 'complete_TUnCaT');
dir_network_input = fullfile(dir_parent,'network_input');

dir_raw_max = 'CNMFE_full mat\raw_max';
if ~exist(dir_raw_max,'dir')
    mkdir(dir_raw_max)
end
dir_SNR_max = 'CNMFE_full mat\SNR_max';
if ~exist(dir_SNR_max,'dir')
    mkdir(dir_SNR_max)
end

for k=1:num_Exp
    clear video_SNR video_raw
    Exp_ID = list_Exp_ID{k};
    
    video_raw=h5read(fullfile(DirData,[Exp_ID,'.h5']), '/mov');
    raw_max = max(video_raw,[],3);
    save(fullfile(dir_raw_max,['raw_max_',Exp_ID,'.mat']),'raw_max');
%     load(['mat\raw_max_',Exp_ID,'.mat'],'raw_max');

    video_SNR=h5read(fullfile(dir_network_input,[Exp_ID,'.h5']), '/network_input');
    SNR_max = max(video_SNR,[],3);
    save(fullfile(dir_SNR_max,['SNR_max_',Exp_ID,'.mat']),'SNR_max');
%     load(['mat\SNR_max_',Exp_ID,'.mat'],'SNR_max');
end
