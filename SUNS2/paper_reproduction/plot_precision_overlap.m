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
    range_overlap = (0:0.05:1)';
elseif data_ind == 2
    dir_sub_SUNS_TUnCaT = 'complete_TUnCaT\4816[1]th4';
    dir_sub_SUNS_FISSA = 'complete_FISSA\4816[1]th3';
%     saved_date_CNMFE = '20220519';
%     saved_date_MIN1PIPE = '20220519';
    saved_date_CNMFE = '20220605';
    saved_date_MIN1PIPE = '20220607';
    range_overlap = (0:0.05:1)';
end

dir_eval_CNMFE = 'C:\Other methods\CNMF_E-1.1.2';
load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb ',saved_date_CNMFE,' cv.mat']),'Table_time_ext');
Table_time_ext_CNMFE = Table_time_ext;
% load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb history ',saved_date,' cv1.mat']),'list_params');

dir_eval_MIN1PIPE = 'C:\Other methods\MIN1PIPE-3.0.0';
load(fullfile(dir_eval_MIN1PIPE,['eval_',data_name,'_thb ',saved_date_MIN1PIPE,' cv.mat']),'Table_time_ext');
Table_time_ext_MIN1PIPE = Table_time_ext;
% load(fullfile(dir_eval_CNMFE,['eval_',data_name,'_thb history ',saved_date,' cv1.mat']),'list_params');
    
Lbin = length(range_overlap)-1;
NGT_overlap = zeros(Lbin,num_Exp);
num_method = 4;
N_overlap = zeros(Lbin,num_method,num_Exp);
Nfind_overlap = zeros(Lbin,num_method,num_Exp);
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
        
        %% Calculate overlap
        area = squeeze(sum(sum(FinalMasks,1),2));
        FinalMasks_sum = sum(FinalMasks,3);
        ncells = length(area);
        overlap = area;
        for n = 1:ncells
            Mask = FinalMasks(:,:,n);
            overlap(n) = sum(sum(Mask & (FinalMasks_sum-Mask)));
        end
        overlap = overlap./area;

        N_find = length(overlap);
        select_overlap = zeros(N_find,Lbin,'logical');
        for ii = 1:Lbin
            select_overlap(:,ii) = (overlap >= range_overlap(ii)) & (overlap < range_overlap(ii+1));
            Nfind_overlap(ii,method,k)=sum(select_overlap(:,ii));
        end

        TP = sum(m,1)>0;
        for ii = 1:Lbin
            in_range = TP(select_overlap(:,ii));
            N_overlap(ii,method,k) = sum(in_range);
        end

    end
end

%%
save(['Precision_overlap ',data_name,' ',num2str(threshJ),'.mat'],'Nfind_overlap','N_overlap');
% %%
% Nfind_overlap_new = Nfind_overlap;
% N_overlap_new = N_overlap;
% load(['Precision_overlap ABO ',num2str(threshJ),'_0921.mat'],'Nfind_overlap','N_overlap');
% Nfind_overlap(:,2,:)=Nfind_overlap_new(:,2,:);
% N_overlap(:,2,:)=N_overlap_new(:,2,:);
% save(['Precision_overlap ABO ',num2str(threshJ),'_0930.mat'],'Nfind_overlap','N_overlap');

%% Precision vs overlap
% load(['Precision_overlap ',data_name,' ',num2str(threshJ),'.mat'],'Nfind_overlap','N_overlap');
Nfind_overlap_all = sum(Nfind_overlap,3);
N_overlap_all = sum(N_overlap,3);
% max_overlap = 60;
max_overlap = max(range_overlap);

bin=1;
% Nfind_overlap_all = sum(reshape(Nfind_overlap_all,bin,[]),1)';
Nfind_overlap_all = reshape(sum(reshape(Nfind_overlap_all,bin,[]),1),[],num_method);
N_overlap_all = reshape(sum(reshape(N_overlap_all,bin,[]),1),[],num_method);
precision_overlap_mean = N_overlap_all./Nfind_overlap_all;

range_overlap_actual = range_overlap;
% range_overlap_actual = 0:bin:max(range_overlap);
range_overlap_show = (range_overlap_actual(1:end-1)+range_overlap_actual(2:end))/2;

figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_overlap_show(1:end),precision_overlap_mean(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),precision_overlap_mean(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),precision_overlap_mean(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),precision_overlap_mean(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Overlap ratio');
ylabel('Precision');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_overlap])

saveas(gcf,['Precision vs overlap ',data_name,' ',num2str(threshJ),'.png'])

%% Number vs overlap
figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_overlap_show(1:end),Nfind_overlap_all(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),Nfind_overlap_all(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),Nfind_overlap_all(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),Nfind_overlap_all(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Overlap ratio');
ylabel('Numbers');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_overlap])

saveas(gcf,['Number vs overlap ',data_name,' ',num2str(threshJ),'.png'])

%% False Positive vs overlap
NFP_overlap_all = Nfind_overlap_all - N_overlap_all;
figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_overlap_show(1:end),NFP_overlap_all(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),NFP_overlap_all(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),NFP_overlap_all(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_overlap_show(1:end),NFP_overlap_all(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Overlap ratio');
ylabel('False positive numbers');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_overlap])
% ylim([0,150]);

saveas(gcf,['FP Number vs overlap ',data_name,' ',num2str(threshJ),'.png'])

