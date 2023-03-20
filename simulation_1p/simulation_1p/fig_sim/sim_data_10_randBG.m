%% initialize working space and prepare for the computation
clear; clc; close all;
addpath('../functions/');
addpath('./extra');
work_dir = fileparts(mfilename('fullpath'));
prepare_env;
noise = 30;
results_folder = ['E:\simulation_CNMFE_randBG\noise',num2str(noise)];
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
    
    %% simulate data
    % cellular signals
    sim_AC;
    ind_center = sub2ind([d1,d2], round(coor.y0), round(coor.x0));
    A = bsxfun(@times, A, reshape(sn(ind_center), 1, []));   %adjust the absolute amplitude neuron signals based on its spatial location
    C = neuron_amp*C;

    % simulate white noise
    E = bsxfun(@times, randn(d1*d2, T), sn);

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

    %% SNR factor = 1
%     Y = A*C + bsxfun(@plus, Bf, Bc) + noise * E;  % observed fluorescence
    Y = A*C + D * F_shuffle + noise * E;  % observed fluorescence
    Y = round(Y); 

    %% reshape to 3D
    Y3 = single(neuron.reshape(Y,2));
    Ysiz = size(Y3);
    A3 = neuron.reshape(A,2);
    FinalMasks = (A3 >= max(max(A3))*0.2);  

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