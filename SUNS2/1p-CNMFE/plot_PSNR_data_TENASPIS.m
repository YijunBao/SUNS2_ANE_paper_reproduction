color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% green = [0.1,0.9,0.1]; % color(5,:); %
% red = [0.9,0.1,0.1]; % color(7,:); %
% blue = [0.1,0.8,0.9]; % color(6,:); %
yellow = [0.8,0.8,0.0]; % color(3,:); %
magenta = [0.9,0.3,0.9]; % color(4,:); %
green = [0.0,0.65,0.0]; % color(5,:); %
red = [0.8,0.0,0.0]; % color(7,:); %
blue = [0.0,0.6,0.8]; % color(6,:); %

% save_figures = true;
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))
addpath('C:\Matlab Files\Filter');
addpath('C:\Matlab Files\Unmixing');

%% neurons and masks frame
data_name = 'TENASPIS';
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
% list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
% rate_hz = [10,15,15,5]; % frame rate of each video
% radius = [5,6,8,14];

doesunmix = 1;
dir_video = 'D:\data_TENASPIS\added_refined_masks';
num_Exp = length(list_Exp_ID);
dir_GT_mask=fullfile(dir_video,'GT Masks');
dir_traces_raw=fullfile(dir_video,'complete_TUnCaT_SF25\TUnCaT\raw');
dir_traces_unmix=fullfile(dir_video,'complete_TUnCaT_SF25\TUnCaT\alpha= 1.000');
% dir_traces = 'D:\ABO\20 percent\GT Masks\';

%%
% dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
% range_SNR = 0:5:100;

% Lbin = length(range_SNR)-1;
% NGT_PSNR = zeros(Lbin,num_Exp);
% num_method = 2;
% N_PSNR = zeros(Lbin,num_method,num_Exp);
% threshJ=0.5;
list_PSNR = cell(1,num_Exp);

%%
for eid=1:num_Exp
    Exp_ID = list_Exp_ID{eid};
%     traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
    if doesunmix
        load(fullfile(dir_traces_unmix,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
        traces = traces_nmfdemix';
        addon = ' unmix';
    else
        load(fullfile(dir_traces_raw,[Exp_ID,'.mat']),'traces','bgtraces'); % raw_traces
        traces = traces - bgtraces;
        traces = traces';
        addon = ' nounmix';
    end
    
    [med, sigma] = SNR_normalization(traces,'quantile-based std');
    SNR = (traces - med)./sigma;
    PSNR = max(SNR,[],2)';
%     med = median(traces,1);
%     sigma = median(abs(traces - med))/(sqrt(2)*erfinv(1/2));
%     SNR = (traces - med)./sigma;
%     PSNR = max(SNR,[],1)';
%     N_GT = length(PSNR);
%     select_SNR = zeros(N_GT,Lbin,'logical');
%     for ii = 1:Lbin
%         select_SNR(:,ii) = (PSNR >= range_SNR(ii)) & (PSNR < range_SNR(ii+1));
%         NGT_PSNR(ii,cv)=sum(select_SNR(:,ii));
%     end
    list_PSNR{eid} = PSNR;

    %%
%     load(fullfile(dir_GT_mask, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
%     FinalMasks = logical(FinalMasks);
%     num = size(FinalMasks,3);
end

%%
PSNR_all = cell2mat(list_PSNR);
mean_PSNR = mean(PSNR_all(~isinf(PSNR_all)));
median_PSNR = median(PSNR_all);
figure;
histogram(PSNR_all,0:5:100);
xlabel('PSNR');
ylabel('Numbers');
text(0.6,0.6,{sprintf('Mean = %.1f',mean_PSNR),sprintf('Median = %.1f',median_PSNR)},...
    'Units','normalized','FontSize',14);
data_name_split = split(data_name,' ');
data_name_short = data_name_split{1};
title([data_name_short,' PSNR distribution'], 'Interpreter','None')
set(gca,'FontSize',14);
saveas(gcf,[data_name_short,' GT Number vs PSNR',addon,'.png'])
disp([mean_PSNR,median_PSNR]);

%%
% [mean(PSNR(1:num)),mean(PSNR((num+1):end))]
