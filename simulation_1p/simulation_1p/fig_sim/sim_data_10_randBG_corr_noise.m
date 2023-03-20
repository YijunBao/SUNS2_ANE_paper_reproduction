%% initialize working space and prepare for the computation
clear; clc; close all;
addpath('../functions/');
addpath('./extra');
work_dir = fileparts(mfilename('fullpath'));
prepare_env;
scale_lowBG = 5e3;
scale_noise = 0.3;
results_folder = sprintf('E:\\simulation_CNMFE_corr_noise\\lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
GT_folder = fullfile(results_folder,'GT Masks');
GT_info = fullfile(results_folder,'GT info');
if ~exist(GT_folder, 'dir')
    mkdir(GT_folder);
end
if ~exist(GT_info, 'dir')
    mkdir(GT_info);
end
%%
for cv = 0:9
    seed = cv;
    filename = ['sim_',num2str(cv)];

    %% load the extracted background and noise from the data
    load('../fig_initialization/data_BG','D','F','sn');
    [d1,d2] = size(sn);
    [rank,T0] = size(F);
    Fs0 = 15;   % frame rate used in the background
    neuron = Sources2D('d1', d1, 'd2', d2);
    sn = neuron.reshape(sn, 1);

    %% parameters for simulating neurons
    rng(seed);
    K = round(200+50*randn); % 200;    % number of neurons
    gSig = 3;   % gaussian width of cell body
    gSiz = 4*gSig+1;
    d_min = 2*gSig;  % minimum distance between neurons
    mu_bar = 0.1;  % mean spike counts per frame
    minIEI = 10;    % minimum interval between events, unit: frame
    minEventSize = 2;  % minimum event size
    T = 2000;   % number of frames
    Fs = 10;     % frame rate
    neuron_amp = 2.0;   % amplify neural signals to modify the signal to noise ratio
    if Fs~=Fs0
        F = imresize(F, [size(F,1), round(size(F,2)*Fs/Fs0)]);
        T0 = size(F,2);
    end
    if T<T0
        multi = ceil(T0/T);
        overlap = ceil((T*multi-T0)/(multi-1));
        F_multi = cell(multi,1);
        for k = 1:multi-1
            F_multi{k} = F(1:end-1,1+(T-overlap)*(k-1):T+(T-overlap)*(k-1));
        end
        F_multi{multi} = F(:,T0-T+1:T0);
        F_multi = cell2mat(F_multi);
    else
        temp = repmat(F, [1, ceil(T/T0)]);
        F_multi = temp(:, 1:T);
    end
%     seed = 2;
    tau_d_bar = 1*Fs;  % mean decay time constant, unit: frame
    tau_r_bar = 0.2*Fs;  % mean rising time constant, unit: frame
    neuron.updateParams('gSig', gSig, 'gSiz', gSiz);
    
    %% cellular signals
    sim_AC;
    ind_center = sub2ind([d1,d2], round(coor.y0), round(coor.x0));
    A = bsxfun(@times, A, reshape(sn(ind_center), 1, []));   %adjust the absolute amplitude neuron signals based on its spatial location
    C = neuron_amp*C;

    %% simulate background
    % background temporal
    select = rand(1,rank)>0.5;
    select(end) = 1;
    rank_multi = size(F_multi,1);
    shuffle = randperm(rank_multi-1);
    shuffle = [shuffle(1:rank-1),rank_multi];
    F_shuffle = select'.*F_multi(shuffle,:);
%     Bc = D*mean(F,2);       % time-invariant baseline
%     Bf = D*bsxfun(@minus, F, mean(F,2));  % fluctuating background

    % background spatial
    rand_flip_lr = rand(1,rank)>0.5;
    rand_flip_ud = rand(1,rank)>0.5;
    for k = 1:rank
        if rand_flip_lr(k)
            D(:,:,k) = fliplr(D(:,:,k));
        end
        if rand_flip_ud(k)
            D(:,:,k) = flipud(D(:,:,k));
        end
    end
    D = neuron.reshape(D, 1);

    % low-passed noise to mimic spatially varying background
    bg_vary = randn(d1,d2,T);
    sigma = [gSiz,gSiz,tau_d_bar]; 
    bg_vary = imgaussfilt3(bg_vary, sigma,'padding','circular');
    bg_vary = neuron.reshape(bg_vary, 1);
    
%     % adjust the magnitude
%     sigma_bg_vary = std(bg_vary,1,'all');
%     bg_invary = D(:,1:end-1) * F_shuffle(1:end-1,:);
%     sigma_bg_invary = std(bg_invary,1,'all');
    
    BG_video = D * F_shuffle + bg_vary * scale_lowBG; % sigma_bg_invary/sigma_bg_vary;
    sn = mean(BG_video,2)/200;
%     BG_mean = reshape(mean(BG_video,2),[d1,d2]);
%     sn = BG_mean;
%     AC_video = A*C;
%     AC_mean = reshape(mean(AC_video,2),[d1,d2]);
%     E_std = reshape(std(E,1,2),[d1,d2]);

    %% simulate noise
%     Y = A*C + bsxfun(@plus, Bf, Bc) + noise * E;  % observed fluorescence
%     Y = A*C + BG_video + noise * E;  % observed fluorescence
    Y_clean = A*C + BG_video;
    Y_pois = poissrnd(Y_clean*scale_noise)/scale_noise;
    noise_sigma = std(Y_pois - Y_clean,1,'all');
%     E = bsxfun(@times, randn(d1*d2, T), sn);
    Y = Y_pois + randn(d1*d2, T)*noise_sigma;    
    Y = round(max(Y,0)); 

    %% reshape to 3D
    Y3 = single(neuron.reshape(Y,2));
    Ysiz = size(Y3);
    A3 = neuron.reshape(A,2);
    FinalMasks = (A3 >= max(max(A3))*0.2);  
    
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
    masks_sum = sum(FinalMasks,3)*0.1;
    contour(masks_sum,'y');

    figure('Position',[700,100,600,400]);
    imagesc((max_Y-min_Y)./mean_Y);
    axis('image');
    colormap gray;
    colorbar;
    hold on;
    contour(masks_sum,'y');

    %% Save video and GT
    % panel_rss;
%     video_mat = fullfile(results_folder, [filename,'.mat']);
%     save(video_mat, 'Y', '-v7.3');
    video_h5 = fullfile(results_folder, [filename,'.h5']);
    if exist(video_h5, 'file')
        delete(video_h5)
    end
    h5create(video_h5,'/mov',Ysiz,'Datatype','single');
    h5write(video_h5,'/mov',Y3);
    
    GT_file = fullfile(GT_info, ['GT_',filename,'.mat']);
    save(GT_file, 'Ysiz', 'D', 'F', 'A', 'C', 'sn', 'A3');
    GT_Masks_file = fullfile(GT_folder, ['FinalMasks_',filename,'.mat']);
    save(GT_Masks_file, 'FinalMasks');
end