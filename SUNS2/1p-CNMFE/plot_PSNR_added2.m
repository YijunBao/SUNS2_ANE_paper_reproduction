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
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
rate_hz = [10,15,7.5,5]; % frame rate of each video
radius = [5,6,6,6];

data_ind = 1;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x,'_added'], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
doesunmix = 1;

for neuron_amp = [0.0005, 0.003, 0.002, 0.001]
sub_added = ['add_neurons_',num2str(neuron_amp),'_rotate']; % ,'_highoverlap'
dir_original = fullfile('E:\data_CNMFE',[data_name,'_original_masks']);
dir_video = fullfile('E:\data_CNMFE',[data_name,'_added_blockwise_weighted_sum_unmask'],sub_added);
% dir_video = 'E:\data_CNMFE\full videos';
dir_GT_original = fullfile(dir_original,'GT Masks');
dir_GT_mask=fullfile(dir_video,'GT Masks');
dir_GT_info=fullfile(dir_video,'GT info');
dir_traces_raw=fullfile(dir_video,'complete_TUnCaT\TUnCaT\raw');
dir_traces_unmix=fullfile(dir_video,'complete_TUnCaT\TUnCaT\alpha= 1.000');
% dir_traces = 'D:\ABO\20 percent\GT Masks\';

%%
dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
range_SNR = 0:2:40;
% range_SNR = 0:5:100;
Lbin = length(range_SNR)-1;
[NGT_PSNR_old, NGT_PSNR_added1, NGT_PSNR_added2] = deal(zeros(Lbin,num_Exp));
[list_PSNR_old,list_PSNR_added1,list_PSNR_added2] = deal(cell(1,num_Exp));
% num_method = 2;
% N_PSNR = zeros(Lbin,num_method,num_Exp);
% threshJ=0.5;

%%
for eid=1:num_Exp
    Exp_ID = list_Exp_ID{eid};
%     traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
    if doesunmix
        load(fullfile(dir_traces_unmix,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
        traces = traces_nmfdemix';
        addon = 'unmix ';
    else
        load(fullfile(dir_traces_raw,[Exp_ID,'.mat']),'traces','bgtraces'); % raw_traces
        traces = traces - bgtraces;
        traces = traces';
        addon = 'nounmix ';
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
%         select_SNR = (PSNR >= range_SNR(ii)) & (PSNR < range_SNR(ii+1));
%         NGT_PSNR(ii,eid)=sum(select_SNR);
%     end

    %%
    load(fullfile(dir_GT_info, ['GT_', Exp_ID, '.mat']), 'masks', 'masks_add');
    masks_add2 = masks_add;
    load(fullfile(dir_GT_original, ['\FinalMasks_', Exp_ID(1:end-6), '.mat']), 'FinalMasks');
    masks_original = FinalMasks;
    masks_add1 = masks(:,:,size(FinalMasks,3)+1:end);
    
    num_old = size(masks_original,3);
    num_added1 = size(masks_add1,3);
    num_added2 = size(masks_add2,3);
    num_1 = size(masks,3);
    PSNR_old = PSNR(1:num_old);
    PSNR_added1 = PSNR((num_old+1):num_1);
    PSNR_added2 = PSNR((num_1+1):end);
    list_PSNR_old{eid} = PSNR_old;
    list_PSNR_added1{eid} = PSNR_added1;
    list_PSNR_added2{eid} = PSNR_added2;
    for ii = 1:Lbin
        select_SNR = (PSNR_old >= range_SNR(ii)) & (PSNR_old < range_SNR(ii+1));
        NGT_PSNR_old(ii,eid)=sum(select_SNR);
        select_SNR = (PSNR_added1 >= range_SNR(ii)) & (PSNR_added1 < range_SNR(ii+1));
        NGT_PSNR_added1(ii,eid)=sum(select_SNR);
        select_SNR = (PSNR_added2 >= range_SNR(ii)) & (PSNR_added2 < range_SNR(ii+1));
        NGT_PSNR_added2(ii,eid)=sum(select_SNR);
    end
    [mean(PSNR_old),mean(PSNR_added1),mean(PSNR_added2)]
end

%%
PSNR_old_sum = sum(PSNR_old,2);
PSNR_added1_sum = sum(PSNR_added1,2);
PSNR_added2_sum = sum(PSNR_added2,2);
PSNR_old_all = cell2mat(list_PSNR_old);
PSNR_added1_all = cell2mat(list_PSNR_added1);
PSNR_added2_all = cell2mat(list_PSNR_added2);
%%
figure('Position',[100,100,400,400]);
histogram(PSNR_old_all,range_SNR, 'FaceColor',yellow);
hold on;
histogram(PSNR_added1_all,range_SNR, 'FaceColor',blue);
histogram(PSNR_added2_all,range_SNR, 'FaceColor',magenta);
legend({'Original neurons','Added real neurons','Added fake neurons'});
xlabel('PSNR');
ylabel('Numbers');
set(gca,'FontSize',14);
title(['Amp = ',num2str(neuron_amp)]);
saveas(gcf,['GT Number vs PSNR added2 ',data_name,'_',sub_added,'.png'])
%%
% for eid = 1:num_Exp
% figure;
% histogram(list_PSNR_old{eid},range_SNR, 'FaceColor',yellow);
% hold on;
% histogram(list_PSNR_added{eid},range_SNR, 'FaceColor',magenta);
% legend({'Original neurons','Added neurons'});
% xlabel('PSNR');
% ylabel('Numbers');
% set(gca,'FontSize',14);
% title(['Amp = ',num2str(neuron_amp),', part ',num2str(eid)]);
% % saveas(gcf,['GT Number vs PSNR added ',data_name,'_',sub_added,'.png'])
% end
%%
end