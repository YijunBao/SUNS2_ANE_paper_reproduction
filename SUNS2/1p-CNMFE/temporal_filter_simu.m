% folder of the GT Masks
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
rate_hz = 10; % frame rate of each video
num_Exp = 9;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);

before=15; % number of frames before spike peak
after=60; % number of frames after spike peak
doesplot=true;
% list_d=[8,10]; % noise1, noise6
% list_d=[5,7]; % noise15
% list_d=[4,5]; % noise15
list_d=[6,8]; % 0.01,1, 5e3,1, & 1e3,1 & 5e3,0.3, & 1e3,0.3   
% list_d=[6,7]; % 1e3,0.1, & 5e3,0.1     
% list_d=[8,10]; % 0.01,10     
% list_d=[3:10]; % two element array showing the minimum and maximum allowed SNR
num_dff=length(list_d)-1;
[array_tau_s,array_tau2_s]=deal(nan(length(list_dataname),num_dff));
spikes_avg_all=nan(length(list_dataname), before+after+1);
time_frame = -before:after;

figure(97);
clf;
set(gcf,'Position',[100,100,500,400]);
hold on;

%%
for vid=1
    %% Load traces and ROIs of all four sub-videos
    dataname = list_dataname{vid};
    dir_masks = fullfile(dir_parent,dataname,'GT Masks');
    dir_trace = fullfile(dir_parent,dataname,'traces');
    fs = rate_hz(vid);
    traces_in = cell(num_Exp,1);
    ROIs = cell(1,1,num_Exp);
    for eid = 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        load(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude');
        traces_in{eid,1} = (traces_raw-traces_bg_exclude);
        load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
        ROIs{1,1,eid}=logical(FinalMasks);
        clear FinalMasks;
    end
    traces_in = cell2mat(traces_in);
    ROIs = cell2mat(ROIs);
    [Lx,Ly,N]=size(ROIs);
    ROIs2 = reshape(ROIs,Lx*Ly,N);

    %% Calculate the average spike shape and decay time
    for ii=1:num_dff
        [tau,tau2,spikes_avg]=determine_decay_time_d_nomix(traces_in, ...
            ROIs2, list_d(ii), list_d(ii+1), before, after, doesplot, fs);
        array_tau_s(vid,ii)=tau/fs;
        array_tau2_s(vid,ii)=tau2/fs;
    end
    spikes_avg_all(vid,:)=spikes_avg;
    
    figure(97);
    plot(time_frame/fs, spikes_avg_all(vid,:), 'LineWidth',2);
end

%% plot the average spike shapes of each video with normalized amplitude
figure(97);
legend(list_dataname);
title('Average Spike Shapes');
xlabel('Time(s)')
% saveas(gcf,'Average Spike Shapes CM.png');
% save('CaImAn spkie template.mat','spikes_avg_all');

%% bar plot of the decay time of each video
figure;
bar(array_tau_s);
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['Decay time from e^{-1} peak']);
xticklabels(list_dataname);
set(gca,'TickLabelInterpreter','None')
% saveas(gcf,['CaImAn tau e-1 peak.png']);

figure;
bar(array_tau2_s);
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['Decay time from 1/2 peak']);
xticklabels(list_dataname);
set(gca,'TickLabelInterpreter','None')
% saveas(gcf,['CaImAn tau 2-1 peak.png']);

%% Save the filter template
for vid=1 % 1:length(list_Exp_ID)
    dataname = list_dataname{vid};
    filter_tempolate = spikes_avg_all(vid,:);
%     save([Exp_ID,'_spike_tempolate.mat'],'filter_tempolate');
    h5_name = [dataname,'_spike_tempolate.h5'];
    if exist(h5_name, 'file')
        delete(h5_name)
    end
    h5create([dataname,'_spike_tempolate.h5'],'/filter_tempolate',[1,1+before+after]);
    h5write([dataname,'_spike_tempolate.h5'],'/filter_tempolate',filter_tempolate);
end

