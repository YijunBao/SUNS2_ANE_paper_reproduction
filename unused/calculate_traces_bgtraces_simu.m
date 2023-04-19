clear;
% folder of the raw video
% dir_parent='E:\simulation_CNMFE_randBG\';
% dir_parent='E:\simulation_CNMFE_corr_noise\';
dir_parent='E:\simulation_constantBG_noise\';
% dir_parent='E:\simulation_CNMFEBG_noise\';
% name of the videos
% scale_lowBG = 5e3;
% scale_noise = 0.3;
% results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
neuron_amp = 0.003;
scale_noise = 1;
results_folder = sprintf('amp=%g,poisson=%g',neuron_amp,scale_noise);
list_dataname={results_folder};
% list_dataname={'noise30'};
num_Exp = 9;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);

%%
for vid= 1
    dataname = list_dataname{vid};
    dir_masks = fullfile(dir_parent,dataname,'GT Masks');
    dir_trace = fullfile(dir_parent,dataname,'traces');
    if ~exist(dir_trace,'dir')
        mkdir(dir_trace);
    end

    for eid = 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        % Load video
        tic;
        fname=fullfile(dir_parent,dataname,[Exp_ID,'.h5']);
        video_raw=h5read(fname, '/mov');
        toc;

        % Load ground truth masks
        load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
        ROIs=logical(FinalMasks);
        clear FinalMasks;

       %%
        tic; 
        traces_raw=generate_traces_from_masks(video_raw,ROIs);
        toc;
        traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_raw,ROIs);
        toc;

        save(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude');
    end
end
