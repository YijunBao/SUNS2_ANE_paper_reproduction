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
doesunmix = 1;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\data_CNMFE',data_name);
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
dir_GT_mask=fullfile(dir_video,'GT Masks');
dir_traces_raw=fullfile(dir_video,'complete_TUnCaT\TUnCaT\raw');
dir_traces_unmix=fullfile(dir_video,'complete_TUnCaT\TUnCaT\alpha= 1.000');
% dir_traces = 'D:\ABO\20 percent\GT Masks\';

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
threshJ=0.5;

%%
for cv=1:num_Exp
    Exp_ID = list_Exp_ID{cv};
    load(fullfile(dir_GT_mask, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
    FinalMasks = logical(FinalMasks);
    area = squeeze(sum(sum(FinalMasks,1),2));
    FinalMasks_sum = sum(FinalMasks,3);
    ncells = length(area);
    overlap = area;
    for n = 1:ncells
        Mask = FinalMasks(:,:,n);
        overlap(n) = sum(sum(Mask & (FinalMasks_sum-Mask)));
    end
    overlap = overlap./area;

    N_GT = length(overlap);
    select_overlap = zeros(N_GT,Lbin,'logical');
    select_overlap(:,1) = (overlap == 0);
    NGT_overlap(1,cv)=sum(select_overlap(:,1));
    for ii = 2:Lbin
        select_overlap(:,ii) = (overlap > range_overlap(ii)) & (overlap <= range_overlap(ii+1));
        NGT_overlap(ii,cv)=sum(select_overlap(:,ii));
    end

    for method = 1:num_method
        %% Load output masks
        switch method
            case 1 % SUNS with TUnCaT
                dir_output_mask = fullfile(dir_video,dir_sub_SUNS_TUnCaT,'output_masks');
                load(fullfile(dir_output_mask, ['Output_Masks_',Exp_ID, '.mat']), 'Masks');
                Masks = permute(Masks,[3,2,1]);

            case 2 % SUNS without FISSA
                dir_output_mask = fullfile(dir_video,dir_sub_SUNS_FISSA,'output_masks');
                load(fullfile(dir_output_mask, ['Output_Masks_',Exp_ID, '.mat']), 'Masks');
                Masks = permute(Masks,[3,2,1]);

            case 3 % CNMF-E
                record = Table_time_ext_CNMFE(cv,:);
                rbg = record(1);
                nk = record(2);
                rdmin = record(3);
                min_corr = record(4);
                min_pnr = record(5);
                merge_thr = record(6);
                mts = record(7);
                mtt = record(8);
                thb = record(9);
                dir_sub_CNMFE = sprintf('gSiz=%d,rbg=%0.1f,nk=%d,rdmin=%0.1f,mc=%0.2f,mp=%d,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
                    gSiz,rbg,nk,rdmin,min_corr,min_pnr,merge_thr,mts,mtt);
                dir_output_mask = fullfile(dir_video,'CNMFE',dir_sub_CNMFE);
                load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
                Masks = logical(Masks3);

            case 4 % MIN1PIPE
                record = Table_time_ext_MIN1PIPE(cv,:);
                pix_select_sigthres = record(1);
                pix_select_corrthres = record(2);
                merge_roi_corrthres = record(3);
                dt = record(4);
                kappa = record(5);
                se = record(6);
                thb = record(7);
%                 dir_sub_MIN1PIPE = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f',...
%                     pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);
                dir_sub_MIN1PIPE = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f_dt=%0.2f_kappa=%0.2f_se=%d',...
                    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa, se);
                dir_output_mask = fullfile(dir_video,'min1pipe',dir_sub_MIN1PIPE);
                load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
                Masks = logical(Masks3);
        end
        
        [Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_mask,Exp_ID,Masks,threshJ);
        TP_2 = sum(m,2)>0;
        for ii = 1:Lbin
            in_range = TP_2(select_overlap(:,ii));
            N_overlap(ii,method,cv) = sum(in_range);
        end
    end
end

save(['recall_overlap ',data_name,' ',num2str(threshJ),'.mat'],'NGT_overlap','N_overlap');
% %%
% NGT_PSNR_new = NGT_PSNR;
% N_PSNR_new = N_PSNR;
% load(['recall_PSNR ABO ',num2str(threshJ),'_0921.mat'],'NGT_PSNR','N_PSNR');
% NGT_PSNR(:,2,:)=NGT_PSNR_new(:,2,:);
% N_PSNR(:,2,:)=N_PSNR_new(:,2,:);
% save(['recall_PSNR ABO ',num2str(threshJ),'_0930.mat'],'NGT_PSNR','N_PSNR');

%% Plot recall vs. PSNR
% load(['recall_PSNR ',data_name,' ',num2str(threshJ),'.mat'],'NGT_PSNR','N_PSNR');
NGT_overlap_all = sum(NGT_overlap,2);
N_overlap_all = sum(N_overlap,3);
% max_overlap = 60;
max_overlap = max(range_overlap);

bin=1;
NGT_overlap_all = sum(reshape(NGT_overlap_all,bin,[]),1)';
N_overlap_all = reshape(sum(reshape(N_overlap_all,bin,[]),1),[],num_method);
recall_overlap_mean = N_overlap_all./NGT_overlap_all;

range_SNR_actual = range_overlap;
% range_SNR_actual = 0:bin:max(range_SNR);
range_SNR_show = (range_SNR_actual(1:end-1)+range_SNR_actual(2:end))/2;

figure('Position',[50,50,500,500],'color','w');
hold on;
plot(range_SNR_show(1:end),recall_overlap_mean(1:end,1),'Color',color(1,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),recall_overlap_mean(1:end,2),'Color',color(2,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),recall_overlap_mean(1:end,3),'Color',color(4,:),'LineWidth',2,'Marker','.','MarkerSize',15);
plot(range_SNR_show(1:end),recall_overlap_mean(1:end,4),'Color',color(5,:),'LineWidth',2,'Marker','.','MarkerSize',15);

xlabel('Overlap ratio');
ylabel('Recall');
legend({'SUNS TUnCaT','SUNS FISSA','CNMF-E','MIN1PIPE'},...
    'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_overlap])

saveas(gcf,['Recall vs overlap ',data_name,' ',num2str(threshJ),'.png'])

%% Plot number of neurons vs. overlap
% inset=axes('Position',[0.6,0.2,0.3,0.3]);
figure('Position',[550,50,500,500],'color','w');
bar(range_SNR_show,NGT_overlap_all); %,'LineWidth',2
xlabel('Overlap ratio');
ylabel('Number of GT neurons');
set(gca,'FontSize',14,'LineWidth',1);
% xlim([0,max_overlap])
% xticks(0:30:60);

saveas(gcf,['GT Number vs overlap ',data_name,' ',num2str(threshJ),'.png'])

