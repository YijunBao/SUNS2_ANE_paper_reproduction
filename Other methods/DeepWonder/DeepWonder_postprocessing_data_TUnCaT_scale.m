% clc, clear
% close all

%% a demo script to show discard too small/too dim components after network segmentation
%  you can supply any other criteria (like throw away boundary components, apply vessel masks, etc)
%  all threshold can be adjusted.
%  last update: 5/30/2020. YZ
addpath(genpath('C:\Matlab Files\missing_finder'));

%%
% target_dir = 'C:\Other methods\DeepWonder\DeepWonder\results\RSM_Tenaspis_20230407-1307';
% filename = 'seg_30_Mouse_1K';
% if 1

% list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
%              'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
% path_name = 'D:\data_TENASPIS\added_refined_masks';
% dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

path_name='E:\1photon-small\added_refined_masks\';
list_Exp_ID = { 'c25_59_228','c27_12_326','c28_83_210',...
                'c25_163_267','c27_114_176','c28_161_149',...
                'c25_123_348','c27_122_121','c28_163_244'};
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

% list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
% list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
% data_ind = 4;
% data_name = list_data_names{data_ind};
% path_name = fullfile('E:\data_CNMFE',data_name);
% dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_
% list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);

num_Exp = length(list_Exp_ID);
[list_Recall, list_Precision, list_F1, list_used_time] = deal(zeros(num_Exp,1));

try
for eid=1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    filename = ['seg_30_',Exp_ID];
    dir_parent = fullfile(path_name,'DeepWonder_scale_full');
    dir_video = fullfile(dir_parent,Exp_ID);
    target_dir = fullfile(dir_video,['DeepWonder_',num2str(patch_size)]);
    dir_GT = fullfile(dir_video,'GT Masks'); % FinalMasks_


    % network_result = importdata(sprintf('%s\\network\\mat\\seg_29_wd.mat', target_dir));
    network_result = load(sprintf('%s\\mat\\%s.mat', target_dir,filename));
    tic;
    % generate A
    N_comp = length(network_result.final_mask_list);
    if N_comp > 0
        network_A = zeros(size(network_result.final_contours, 1), size(network_result.final_contours, 2), N_comp);
        network_C = [];
        network_A_center = [];
        for i = 1 : N_comp
            buf = zeros(size(network_A, 1), size(network_A, 2));
            valid_ind = sub2ind(size(buf), network_result.final_mask_list{i}.position(:, 1) + 1, network_result.final_mask_list{i}.position(:, 2) + 1);
            buf(valid_ind) = 1;
            network_A(:, :, i) = buf;
            network_C = [network_C; network_result.final_mask_list{i}.trace];
        end
        % throw away components that are too small
        area_threshold = 50;
        invalid_ind = [];
        for i = 1 : N_comp
            buf = network_A(:, :, i);
            % calculate area
            curr_area = sum(buf, 'all');

            if curr_area < area_threshold
                invalid_ind(i) = 1;
            else
                invalid_ind(i) = 0;
            end
        end
        network_A(:, :, find(invalid_ind)) = [];
        network_C(find(invalid_ind), :) = [];


        %%
        network_A_readout = [];
        buf = dir(sprintf('%s\\RMBG\\*.tif', target_dir));
        network_raw = loadtiff(sprintf('%s\\\\RMBG\\%s', target_dir, buf(1).name));
        network_raw = single(network_raw);

        for i = 1 : size(network_A ,3)
        %     i
            buf = network_A(:, :, i);
            curr_net_sig = squeeze(mean(bsxfun(@times, network_raw, buf), [1, 2]));

            network_A_readout(i, :) = curr_net_sig;
        end
        %% throw away those too dim neurons
        threshold = 0.05;
        std_network_A_readout = std(network_A_readout, 0, 2);
        max_val = max(std_network_A_readout);
        valid_ind = std_network_A_readout > max_val * threshold;
        valid_ind = find(valid_ind);

        %% output filtered components
        network_A_center = com(reshape(network_A, [], size(network_A, 3)), size(network_A, 1), size(network_A, 2));
        network_A_center_filt = network_A_center(valid_ind, :);
        network_A_filt = network_A(:, :, valid_ind);
        network_C_filt = network_C(valid_ind, :);

    %%
    else
        network_A_filt = [];
        network_C_filt = [];
    end
        
    post_time = toc;
    total_time = network_result.used_time + post_time;
    figure; imagesc(sum(network_A_filt,3)); colorbar; axis image;
    [Recall, Precision, F1] = GetPerformance_Jaccard(dir_GT,Exp_ID,permute(network_A_filt,[2,1,3]),0.5);
    save(sprintf('%s\\mat\\%s.mat', target_dir,[filename,'_post']),...
        'network_A_filt','network_C_filt','Recall','Precision','F1','total_time')
    list_Recall(eid) = Recall;
    list_Precision(eid) = Precision;
    list_F1(eid) = F1;
    list_used_time(eid) = total_time;
end

Table_time = [list_Recall, list_Precision, list_F1, list_used_time];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
save(fullfile(dir_parent,['eval_',num2str(patch_size),'.mat']),...
    'list_Recall','list_Precision','list_F1','list_used_time','Table_time_ext')
end

%%
function cm = com(A,d1,d2,d3)
% cm short for center of mass
% center of mass calculation
% inputs:
% A: d X nr matrix, each column in the spatial footprint of a neuron
% d1, d2, d3: the dimensions of the 2-d (or 3-d) field of view

% output:
% cm: nr x 2 (or 3) matrix, with the center of mass coordinates

    if nargin < 4
        d3 = 1;
    end
    if d3 == 1
        ndim = 2;
    else
        ndim = 3;
    end

    nr = size(A,2);
    Coor.x = kron(ones(d2*d3,1),double(1:d1)');
    Coor.y = kron(ones(d3,1),kron(double(1:d2)',ones(d1,1)));
    Coor.z = kron(double(1:d3)',ones(d2*d1,1));
    cm = [Coor.x, Coor.y, Coor.z]'*A/spdiags(sum(A)',0,nr,nr);
    cm = cm(1:ndim,:)';
end
