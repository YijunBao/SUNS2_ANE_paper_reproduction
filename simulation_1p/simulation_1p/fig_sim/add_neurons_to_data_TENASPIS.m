addpath('../functions/');
addpath('./extra');
work_dir = fileparts(mfilename('fullpath'));
prepare_env;

%%
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
rate_hz = 20; % frame rate of each video
radius = 9;

%%
neuron_amp = 0.003;
num_Exp = length(list_Exp_ID);
gSig = round(radius/2); % 3;
gSiz = 4*gSig+1;
d_min = 1.5*gSig;  % minimum distance between neurons
Fs = rate_hz;     % frame rate
minIEI = Fs;    % minimum interval between events, unit: frame
mu_bar = 1/Fs;  % mean spike counts per frame
minEventSize = 2;  % minimum event size
tau_d_bar = 4;  % mean decay time constant, unit: frame
tau_r_bar = 2;  % mean rising time constant, unit: frame
% tau_d_bar = 1*Fs;  % mean decay time constant, unit: frame
% tau_r_bar = 0.2*Fs;  % mean rising time constant, unit: frame
elong = 0.3; % elongation 

%%
dir_video='D:\data_TENASPIS\added_refined_masks';
% dir_video=fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise_weighted_sum_unmask']);
dir_masks = fullfile(dir_video, 'GT Masks');
% dir_trace = fullfile(dir_video, 'complete_TUnCaT\TUnCaT\alpha= 1.000');
dir_add_new = fullfile(dir_video, ['add_neurons_',num2str(neuron_amp),'_rotate']);
GT_folder = fullfile(dir_add_new,'GT Masks');
GT_info = fullfile(dir_add_new,'GT info');
if ~ exist(dir_add_new,'dir')
    mkdir(dir_add_new);
end
if ~exist(GT_folder, 'dir')
    mkdir(GT_folder);
end
if ~exist(GT_info, 'dir')
    mkdir(GT_info);
end

%%
for eid = 1:num_Exp
    %% Load video and masks
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    masks=logical(FinalMasks); % permute(logical(Masks),[3,2,1]);
    fname=fullfile(dir_video,[Exp_ID,'.h5']);
    video_raw = h5read(fname, '/mov');
    [d1,d2,T] = size(video_raw);
    sn = mean(video_raw,3);
    filename = [Exp_ID, '_added'];
    
    %% Set parameters
    seed = eid;
    K = round(200+10*randn);      % number of neurons

    %% Simulate neurons
    neuron = Sources2D('d1', d1, 'd2', d2);
    neuron.updateParams('gSig', gSig, 'gSiz', gSiz);
    sim_AC; 
    ind_center = sub2ind([d1,d2], round(coor.y0), round(coor.x0));
    A = bsxfun(@times, A, reshape(sn(ind_center), 1, []));   %adjust the absolute amplitude neuron signals based on its spatial location
    Ab = A > 0.2*max(A,[],1);   %adjust the absolute amplitude neuron signals based on its spatial location
    Ab3 = neuron.reshape(Ab, 2);   %adjust the absolute amplitude neuron signals based on its spatial location
    C = neuron_amp*C;
    
    %% Remove neurons with high overlap
    masks_sum_original = sum(masks,3);
    masks_sum = masks_sum_original;
    num_add = size(A,2);
    list_add = false(1,num_add);
    for n = 1:num_add
        mask = Ab3(:,:,n);
        if max(masks_sum(mask)) <= 2
            list_add(n) = true;
            masks_sum = masks_sum + mask;
        end
    end
    
    %% Add neurons to video
    masks_add = Ab3(:,:,list_add);
    video_add = neuron.reshape(A(:,list_add) * C(list_add,:), 2);
    Y3 = video_raw + (video_add);
    FinalMasks = cat(3, masks, masks_add);
    
    %% Show video and GT
    figure('Position',[100,100,600,400]);
    max_Y = max(Y3,[],3);
    min_Y = min(Y3,[],3);
    mean_Y = mean(Y3,3);
    imagesc(max_Y);
    axis('image');
    colormap gray;
    colorbar;
    hold on;
    contour(masks_sum*0.01,'y');
    masks_add_sum = sum(masks_add,3)*0.01;
    contour(masks_add_sum,'g');

    figure('Position',[700,100,600,400]);
    imagesc(double(max_Y-min_Y)./mean_Y);
    axis('image');
    colormap gray;
    colorbar;
    hold on;
    contour(masks_sum*0.01,'y');
    masks_add_sum = sum(masks_add,3)*0.01;
    contour(masks_add_sum,'g');

    %% Save results
    video_h5 = fullfile(dir_add_new, [filename,'.h5']);
    if exist(video_h5, 'file')
        delete(video_h5)
    end
    Ysiz = size(Y3); 
    h5create(video_h5,'/mov',Ysiz,'Datatype',class(video_raw));
    h5write(video_h5,'/mov',Y3);
    
    GT_file = fullfile(GT_info, ['GT_',filename,'.mat']);
    save(GT_file,'masks', 'masks_add', 'Ysiz', 'A', 'C', 'sn', 'list_add');
    GT_Masks_file = fullfile(GT_folder, ['FinalMasks_',filename,'.mat']);
    save(GT_Masks_file, 'FinalMasks');
end

