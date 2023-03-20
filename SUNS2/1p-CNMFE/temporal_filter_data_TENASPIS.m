% folder of the GT Masks
dir_parent='D:\data_TENASPIS\added_refined_masks\';
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp_ID = length(list_Exp_ID);
rate_hz = 20; % frame rate of each video
         
before=15; % number of frames before spike peak
after=60; % number of frames after spike peak
doesplot=false;
% list_d=[6,7]; % blood_vessel_10Hz
% list_d=[6,7]; % PFC4_15Hz
% list_d=[10,12]; % PFC4_15Hz
list_d=[8:2:16]; % two element array showing the minimum and maximum allowed SNR
num_dff=length(list_d)-1;
num_Exp = length(list_Exp_ID);
[array_tau_s,array_tau2_s,num_spikes]=deal(nan(num_Exp, num_dff));
spikes_avg_all=nan(num_Exp, num_dff, before+after+1);
time_frame = -before:after;

for ii=1:num_dff
    figure(90+ii);
    clf;
    set(gcf,'Position',[100,100,500,400]);
    hold on;
end

%%
traces_in = cell(num_Exp,1);
ROIs = cell(1,1,num_Exp);
ROIs2 = cell(1,1,num_Exp);
for vid=1:num_Exp_ID
    %% Load traces and ROIs of all four sub-videos
    Exp_ID = list_Exp_ID{vid};
    dir_masks = fullfile(dir_parent,'GT Masks');
    dir_trace = fullfile(dir_parent,'traces');
    fs = rate_hz;
    load(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude');
    traces_in{vid,1} = (traces_raw-traces_bg_exclude);
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    ROIs{1,1,vid}=logical(FinalMasks);
    [Lx,Ly,N]=size(FinalMasks);
    ROIs2{1,1,vid} = reshape(ROIs{1,1,vid},Lx*Ly,N);
end
% traces_in = cell2mat(traces_in);
% ROIs = cell2mat(ROIs);
% [Lx,Ly,N]=size(ROIs);
% ROIs2 = reshape(ROIs,Lx*Ly,N);

%% Calculate the average spike shape and decay time
for ii=1:num_dff
    for vid=1:num_Exp_ID
        [tau,tau2,spikes_avg,list_spikes_all]=determine_decay_time_d_nomix(traces_in{vid,1}, ...
            ROIs2{1,1,vid}, list_d(ii), list_d(ii+1), before, after, doesplot, fs);
        array_tau_s(vid,ii)=tau/fs;
        array_tau2_s(vid,ii)=tau2/fs;
        num_spikes(vid,ii)=size(list_spikes_all,1);
        spikes_avg_all(vid,ii,:)=spikes_avg;
        figure(90+ii);
        plot(time_frame/fs, squeeze(spikes_avg_all(vid,ii,:)), 'LineWidth',2);
    end
    legend(list_Exp_ID,'Interpreter','None');
    title('Average Spike Shapes');
    xlabel('Time(s)')
end

%%
list_legend = arrayfun(@(x,y) [num2str(x),'<d<',num2str(y)],...
    list_d(1:end-1),list_d(2:end),'UniformOutput',false);

%% bar plot of the decay time of each video
figure;
bar(array_tau_s);
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['TENASPIS videos, Decay time from e^{-1} peak']);
xticklabels(list_Exp_ID); % ,'Interpreter','None'
legend(list_legend);
set(gca,'TickLabelInterpreter','None')
saveas(gcf,['TENASPIS tau e-1 each.png']);

figure;
bar(array_tau2_s);
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['TENASPIS videos, Decay time from 1/2 peak']);
xticklabels(list_Exp_ID); % ,'Interpreter','None'
legend(list_legend);
set(gca,'TickLabelInterpreter','None')
saveas(gcf,['TENASPIS tau 2-1 each.png']);

%% bar plot of the decay time of all videos
figure;
bar(mean(array_tau_s,1));
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['TENASPIS videos, Decay time from e^{-1} peak']);
xticklabels(list_legend);
set(gca,'TickLabelInterpreter','None')
saveas(gcf,['TENASPIS tau e-1 all.png']);

figure;
bar(mean(array_tau2_s,1));
xlabel('Video id');
ylabel('Decay time, \tau (s)');
title(['TENASPIS videos, Decay time from 1/2 peak']);
xticklabels(list_legend);
set(gca,'TickLabelInterpreter','None')
saveas(gcf,['TENASPIS tau 2-1 all.png']);

%% Save the filter template
ii_select = 2;
filter_tempolate = squeeze(mean(spikes_avg_all(:,ii_select,:),1))';
save(['TENASPIS_spike_tempolate.mat'],'filter_tempolate');
h5_name = ['TENASPIS_spike_tempolate.h5'];
if exist(h5_name, 'file')
    delete(h5_name)
end
h5create(['TENASPIS_spike_tempolate.h5'],'/filter_tempolate',[1,1+before+after]);
h5write(['TENASPIS_spike_tempolate.h5'],'/filter_tempolate',filter_tempolate);

%%
save('TENASPIS_template_data.mat','array_tau_s','array_tau2_s','list_d','spikes_avg_all','num_spikes','ii_select');
