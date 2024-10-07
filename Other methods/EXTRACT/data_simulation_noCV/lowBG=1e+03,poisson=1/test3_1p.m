addpath(genpath('C:\Users\Yijun\Documents\MATLAB\EXTRACT-public-master'));
%% Start the main pipeline
filenames = {'sim_0','sim_1','sim_2','sim_3','sim_4','sim_5','sim_6','sim_7','sim_8','sim_9'};
radii = 8;
SNRs = 3;
output_folder = ['./','EXTRACT_radius',num2str(radii),'_SNR',num2str(SNRs)','./'];
if ~exist(output_folder,'dir')
    mkdir(output_folder);
end

CV_times = zeros(length(filenames),1);
for p = 1:length(filenames)
    output_name = [output_folder,filenames{p},'_EXTRACT.mat'];
%     if ~isfile(output_name)

hinfo = h5info(['../' filenames{p} '.h5']);
nx = hinfo.Datasets.Dataspace.Size(1);
ny = hinfo.Datasets.Dataspace.Size(2);
totalnum = hinfo.Datasets.Dataspace.Size(3);

%% Downsample the movie
% downsampletime_pipeline('~/data1/test2/SUNS2_ANE_paper_reproduction-main/data/data_TENASPIS/added_refined_masks/Mouse_4K.h5:/mov',1,1,1590)
% Downsamples the first 40000 frames of the movie by 4 using 40 blocks. You can downsample the movie down to 2Hz, maybe even more...


%% run EXTRACT on the downsampled movie

M = h5read(['../' filenames{p} '.h5'],'/mov');
config =[]
config = get_defaults(config);
tic
config.avg_cell_radius=radii;
config.num_partitions_x=1;
config.num_partitions_y=1;

% change these as needed
config.cellfind_min_snr = 1;
config.thresholds.T_min_snr=SNRs;

output=extractor(M,config);
% save([output_folder,'extract_downsampled_unsorted.mat'],'output','-v7.3');

%% Cell sorting

% While it is optional, it is beneficial to sort the cells at this stage before moving forward.
%cell_check(output, M)

%% run EXTRACT on the full movie 
% load([output_folder,'extract_downsampled_unsorted.mat']);
%M = ['../' filenames{p} '.h5:/mov'];
config = output.config;

config.avg_cell_radius=10;

% Add more partitions as needed for the RAM memory. As a rule of thumb, you want to partition the movie such that partitioned movie memory is 1/4th of RAM memory.
config.num_partitions_x=1;
config.num_partitions_y=1;

config.max_iter=0;

% If you sorted, make sure that S_in is the sorted cell filters coming from cell check above!
S_in=output.spatial_weights;
config.S_init=full(reshape(S_in, size(S_in, 1) * size(S_in, 2), size(S_in, 3)));

output=extractor(M,config);
CV_times(p) = toc;
save(output_name,'output','-v7.3');
%     end
%     CV_times
end
save([output_folder,'noCV_times.mat'], 'CV_times');