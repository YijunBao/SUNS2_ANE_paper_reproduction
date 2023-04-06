clear;
% warning off

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

data_ind = 2;
doesunmix = 1;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\data_CNMFE',data_name);
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
dir_GT_mask=fullfile(dir_video,'GT Masks');
% dir_traces = 'D:\ABO\20 percent\GT Masks\';
list_folders = {'complete_TUnCaT','complete_FISSA','CNMFE','min1pipe'};

%%
gSiz = 2*radius(data_ind); 
if data_ind == 1
    dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
    dir_sub_SUNS_FISSA = 'complete_FISSA\4816[1]th4';
%     saved_date_CNMFE = '20220521';
%     saved_date_MIN1PIPE = '20220522';
    saved_date_CNMFE = '20220605';
    saved_date_MIN1PIPE = '20220606';
    range_SNR = (0.5:0.05:1.2)';
elseif data_ind == 2
    dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
    dir_sub_SUNS_FISSA = 'complete_FISSA\4816[1]th3';
%     saved_date_CNMFE = '20220519';
%     saved_date_MIN1PIPE = '20220519';
    saved_date_CNMFE = '20220605';
    saved_date_MIN1PIPE = '20220607';
    range_SNR = (0.5:0.05:1.2)';
end

dir_eval_CNMFE = 'C:\Other methods\CNMF_E-1.1.2';
load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb ',saved_date_CNMFE,' cv.mat']),'Table_time_ext');
Table_time_ext_CNMFE = Table_time_ext;
% load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb history ',saved_date,' cv1.mat']),'list_params');

dir_eval_MIN1PIPE = 'C:\Other methods\MIN1PIPE-3.0.0';
load(fullfile(dir_eval_MIN1PIPE,['eval_',data_name,'_thb ',saved_date_MIN1PIPE,' cv.mat']),'Table_time_ext');
Table_time_ext_MIN1PIPE = Table_time_ext;
% load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb history ',saved_date,' cv1.mat']),'list_params');
    
Lbin = length(range_SNR)-1;
NGT_PSNR = zeros(Lbin,num_Exp);
num_method = 4;
N_PSNR = zeros(Lbin,num_method,num_Exp);
Nfind_PSNR = zeros(Lbin,num_method,num_Exp);
threshJ=0.5;

%% Load video
for k=1:num_Exp
    Exp_ID=list_Exp_ID{k};
    for method = 1:num_method
        folder = list_folders{method};
        %% Load output masks
        dir_output_mask = fullfile(dir_video,folder,'output_masks');
        load(fullfile(dir_output_mask, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
        [Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_mask,Exp_ID,FinalMasks,threshJ);
        
        %% Calculate PSNR
        dir_traces_raw=fullfile(dir_output_mask,'TUnCaT\raw');
        dir_traces_unmix=fullfile(dir_output_mask,'TUnCaT\alpha= 1.000');
        load(fullfile(dir_traces_raw,[Exp_ID,'.mat']),'traces','bgtraces'); % raw_traces
        sigma_bg = std(bgtraces,1,1);
        PSNR = sigma_bg';
%         if doesunmix
%             load(fullfile(dir_traces_unmix,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
%             traces = traces_nmfdemix';
%             addon = 'unmix ';
%         else
%             traces = traces - bgtraces;
%             traces = traces';
%             addon = 'nounmix ';
%         end
    
%         [med, sigma] = SNR_normalization(traces,'quantile-based std');
%         SNR = (traces - med)./sigma;
%         PSNR = max(SNR,[],2)';
        
        N_find = length(PSNR);
        select_SNR = zeros(N_find,Lbin,'logical');
        for ii = 1:Lbin
            select_SNR(:,ii) = (PSNR >= range_SNR(ii)) & (PSNR < range_SNR(ii+1));
            Nfind_PSNR(ii,method,k)=sum(select_SNR(:,ii));
        end

        TP = sum(m,1)>0;
        for ii = 1:Lbin
            in_range = TP(select_SNR(:,ii));
            N_PSNR(ii,method,k) = sum(in_range);
        end

    end
end

%%
save(['Precision_sigma_bg ',data_name,' ',num2str(threshJ),'.mat'],'Nfind_PSNR','N_PSNR');
% %%
% Nfind_PSNR_new = Nfind_PSNR;
% N_PSNR_new = N_PSNR;
% load(['Precision_PSNR ABO ',num2str(threshJ),'_0921.mat'],'Nfind_PSNR','N_PSNR');
% Nfind_PSNR(:,2,:)=Nfind_PSNR_new(:,2,:);
% N_PSNR(:,2,:)=N_PSNR_new(:,2,:);
% save(['Precision_PSNR ABO ',num2str(threshJ),'_0930.mat'],'Nfind_PSNR','N_PSNR');

%% Precision vs PSNR
load(['Precision_sigma_bg ',data_name,' ',num2str(threshJ),'.mat'],'Nfind_PSNR','N_PSNR');
Nfind_PSNR_all = sum(Nfind_PSNR,3);
N_PSNR_all = sum(N_PSNR,3);
% max_PSNR = 60;
max_PSNR = max(range_SNR);

bin=1;
% Nfind_PSNR_all = sum(reshape(Nfind_PSNR_all,bin,[]),1)';
Nfind_PSNR_all = reshape(sum(reshape(Nfind_PSNR_all,bin,[]),1),[],num_method);
N_PSNR_all = reshape(sum(reshape(N_PSNR_all,bin,[]),1),[],num_method);
precision_PSNR_mean = N_PSNR_all./Nfind_PSNR_all;

range_SNR_actual = range_SNR;
% range_SNR_actual = 0:bin:max(range_SNR);
range_SNR_show = (range_SNR_actual(1:end-1)+range_SNR_actual(2:end))/2;

figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_SNR_show(1:end),precision_PSNR_mean(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),precision_PSNR_mean(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),precision_PSNR_mean(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),precision_PSNR_mean(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Background variation');
ylabel('Precision');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_PSNR])

saveas(gcf,['Precision vs sigma_bg ',data_name,' ',num2str(threshJ),'.png'])

%% Number vs PSNR
figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_SNR_show(1:end),Nfind_PSNR_all(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),Nfind_PSNR_all(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),Nfind_PSNR_all(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),Nfind_PSNR_all(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Background variation');
ylabel('Numbers');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_PSNR])

saveas(gcf,['Number vs PSNR ',data_name,' ',num2str(threshJ),'.png'])

%% False Positive vs PSNR
NFP_PSNR_all = Nfind_PSNR_all - N_PSNR_all;
figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_SNR_show(1:end),NFP_PSNR_all(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),NFP_PSNR_all(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),NFP_PSNR_all(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),NFP_PSNR_all(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Background variation');
ylabel('False positive numbers');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_PSNR])
% ylim([0,150]);

saveas(gcf,['FP Number vs sigma_bg ',data_name,' ',num2str(threshJ),'.png'])

