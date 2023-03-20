addpath('C:\Matlab Files\timer');
timer_start_next;
try
% CNMFE_simu_direvl_thb_cv_test;
% CNMFE_simu_init_for_par;
CNMFE_par_simu_direvl_thb;
% CNMFE_par_simu_direvl_thb_cv;
% CNMFE_par_data_CNMFE_add;
% CNMFE_par_data_TENASPIS_direvl_thb;
% CNMFE_data_TENASPIS_direvl_test;
% CNMFE_par_data_CNMFE_direvl_thb;
% CNMFE_par_data_CNMFE_direvl_thb_cv;
end
timer_stop;

%%
pre_CNN_data_CNMFE_crop_SUNS_blockwise;
Copy_of_pre_CNN_data_CNMFE_crop_SUNS_blockwise;
Copy_2_of_pre_CNN_data_CNMFE_crop_SUNS_blockwise;
add_weights_data_CNMFE_crop_SUNS_blockwise;
Copy_of_add_weights_data_CNMFE_crop_SUNS_blockwise;
Copy_2_of_add_weights_data_CNMFE_crop_SUNS_blockwise;


%%
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
green = [0.1,0.9,0.1];
red = [0.9,0.1,0.1];
blue = [0.1,0.8,0.9];
yellow = [0.9,0.9,0.1];
magenta = [0.9,0.3,0.9];

%% Show optimal parameters from F1All (CV).mat for CaImAn dataset
% load('eval_1p_small 1x3x1x3x3x3.mat');
% load('eval_1p_small.mat');

% min_corr_default = 0.8;
% min_pnr_default = 8;
% merge_thr_default = 0.6; % 0.65;
% merge_thr_spatial_default = 0.8;
% merge_thr_temporal_default = 0.4;
% nk_default = 3;
% ind1_default = find(list_nk == nk_default);
% ind2_default = find(list_min_corr == min_corr_default);
% ind3_default = find(list_min_pnr == min_pnr_default);
% ind4_default = find(list_merge_thr == merge_thr_default);
% ind5_default = find(list_merge_thr_spatial == merge_thr_spatial_default);
% ind6_default = find(list_merge_thr_temporal == merge_thr_temporal_default);

figure('Position',[0,100,1520,900]);
num_Exp = size(list_F1,1);

shape_F1 = size(list_F1);
F1mean = reshape(mean(list_F1(:,:,:,:,:,:,:),1),shape_F1(2:end));
Recallmean = reshape(mean(list_Recall(:,:,:,:,:,:,:),1),shape_F1(2:end));
Precisionmean = reshape(mean(list_Precision(:,:,:,:,:,:,:),1),shape_F1(2:end));
[F1_max, ~]=max(F1mean(:));
list_ind_max = find(F1mean==F1_max);
length_max = length(list_ind_max);
table_ind_max = zeros(length_max,6);
for ii = 1:length_max
    [ind1, ind2, ind3, ind4, ind5, ind6]=ind2sub(size(F1mean),list_ind_max(ii));
    table_ind_max(ii,:) = [ind1, ind2, ind3, ind4, ind5, ind6];
end
table_ind_max_sum = sum(table_ind_max,2);
[~,ii] = max(table_ind_max_sum);
[ind1, ind2, ind3, ind4, ind5, ind6]=ind2sub(size(F1mean),list_ind_max(ii));
rdmin = list_rdmin(ind1);
min_corr = list_min_corr(ind2);
min_pnr = list_min_pnr(ind3);
merge_thr = list_merge_thr(ind4);
merge_thr_spatial = list_merge_thr_spatial(ind5);
merge_thr_temporal = list_merge_thr_temporal(ind6);
% F1_max = F1_max;
Recall_max = Recallmean(list_ind_max(ii));
Precision_max = Precisionmean(list_ind_max(ii));
list_time_opt = squeeze(list_time(:, ind1, ind2, ind3, ind4, ind5, ind6));
list_Recall_max = squeeze(list_Recall(:, ind1, ind2, ind3, ind4, ind5, ind6));
list_Precision_max = squeeze(list_Precision(:, ind1, ind2, ind3, ind4, ind5, ind6));
list_F1_max = squeeze(list_F1(:, ind1, ind2, ind3, ind4, ind5, ind6));
% F1_default = F1mean(ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
% Recall_default = Recallmean(ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
% Precision_default = Precisionmean(ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);

subplot(2,3,1);    hold on;
plot(list_rdmin,squeeze(F1mean(:,ind2,ind3,ind4,ind5,ind6)),'LineWidth',2);
plot(list_rdmin(squeeze(F1mean(:,ind2,ind3,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(list_nk,squeeze(F1mean(:,ind2,ind3,ind4,ind5,ind6)),'LineWidth',2);
% plot(list_nk(squeeze(F1mean(:,ind2,ind3,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(nk_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(2,3,2);    hold on;
plot(list_min_corr,squeeze(F1mean(ind1,:,ind3,ind4,ind5,ind6)),'LineWidth',2);
plot(list_min_corr(squeeze(F1mean(ind1,:,ind3,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(min_corr_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(2,3,3);    hold on;
% plot(list_rdmin,squeeze(F1mean(ind1,ind2,:,ind4,ind5,ind6)),'LineWidth',2);
% plot(list_rdmin(squeeze(F1mean(ind1,ind2,:,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
plot(list_min_pnr,squeeze(F1mean(ind1,ind2,:,ind4,ind5,ind6)),'LineWidth',2);
plot(list_min_pnr(squeeze(F1mean(ind1,ind2,:,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(min_pnr_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(2,3,4);    hold on;
plot(list_merge_thr,squeeze(F1mean(ind1,ind2,ind3,:,ind5,ind6)),'LineWidth',2);
plot(list_merge_thr(squeeze(F1mean(ind1,ind2,ind3,:,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_thr_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(2,3,5);    hold on;
plot(list_merge_thr_spatial,squeeze(F1mean(ind1,ind2,ind3,ind4,:,ind6)),'LineWidth',2);
plot(list_merge_thr_spatial(squeeze(F1mean(ind1,ind2,ind3,ind4,:,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_thr_spatial_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(2,3,6);    hold on;
plot(list_merge_thr_temporal,squeeze(F1mean(ind1,ind2,ind3,ind4,ind5,:)),'LineWidth',2);
plot(list_merge_thr_temporal(squeeze(F1mean(ind1,ind2,ind3,ind4,ind5,:))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_thr_temporal_default,F1_default,'*','Color',green,'LineWidth',2);

% disp(['F1_max = ',replace(num2str(list_F1_max),'     ',', ')])
% disp(['rval_thr = ',replace(mat2str(list_rval_thr),' ',', ')])
% disp(['min_SNR = ',replace(mat2str(list_min_SNR),' ',', ')])
% disp(['cnn_thr = ',replace(mat2str(list_cnn_thr),' ',', ')])
% disp(['cnn_lowest = ',replace(mat2str(list_cnn_lowest),' ',', ')])
Table_eval=[list_Recall_max, list_Precision_max, list_F1_max, list_time_opt]; 
Table_params = [rdmin, min_corr, min_pnr, merge_thr, merge_thr_spatial, merge_thr_temporal];
Table = [repmat(Table_params,[4,1]), Table_eval];
Table_ext = [Table; mean(Table,1); std(Table,1,1)];

clear ax
ax(1)=subplot(2,3,1);
% set(gca,'Fontsize',12);
% expID = cellfun(@num2str, (num2cell(0:3)), 'UniformOutput',false);
% legend(expID,'Location','South','NumColumns',2)
ylabel('{\itF_1}');
xlabel('rdmin','Interpreter','None');
% xlabel('nk','Interpreter','None');
ax(2)=subplot(2,3,2);
% set(gca,'Fontsize',12);
xlabel('min_corr','Interpreter','None');
ax(3)=subplot(2,3,3);
% set(gca,'Fontsize',12);
xlabel('min_pnr','Interpreter','None');
% xlabel('min_pnr','Interpreter','None');
ax(4)=subplot(2,3,4);
% set(gca,'Fontsize',12);
xlabel('merge_thr','Interpreter','None');
ax(5)=subplot(2,3,5);
% set(gca,'Fontsize',12);
xlabel('merge_thr_spatial','Interpreter','None');
ax(6)=subplot(2,3,6);
% set(gca,'Fontsize',12);
xlabel('merge_thr_temporal','Interpreter','None');
linkaxes(ax,'y');
set(ax,'Fontsize',12);
ylim([0.3,0.45]);
% suptitle('CaImAn batch parameter dependence')

% saveas(gcf,'CNMFE_mean_F1 vs params 1p_small.png')

%% Show optimal parameters from cross validation
shape_F1 = size(list_F1);
shape_F1 = padarray(shape_F1,[0,7-length(shape_F1)],1,'post');
num_Exp = shape_F1(1);
[list_F1_CV,list_Recall_CV,list_Precision_CV,list_time_CV,list_F1_default,list_Recall_default,list_Precision_default,...
    opt_rdmin,opt_min_corr,opt_min_pnr,opt_merge_thr,opt_merge_thr_spatial,opt_merge_thr_temporal]=deal(zeros(num_Exp,1));

for CV = 1:num_Exp
    train = setdiff(1:num_Exp,CV);
    F1mean = reshape(mean(list_F1(train,:,:,:,:,:,:),1),shape_F1(2:end));
%     Recallmean = reshape(mean(list_Recall(train,:,:,:,:,:,:),1),shape_F1(2:end));
%     Precisionmean = reshape(mean(list_Precision(train,:,:,:,:,:,:),1),shape_F1(2:end));
    [F1_max, ~]=max(F1mean(:));
    list_ind_max = find(F1mean==F1_max);
    length_max = length(list_ind_max);
    table_ind_max = zeros(length_max,length(shape_F1)-1);
    for ii = 1:length_max
        [ind1, ind2, ind3, ind4, ind5, ind6]=ind2sub(size(F1mean),list_ind_max(ii));
        table_ind_max(ii,:) = [ind1, ind2, ind3, ind4, ind5, ind6];
    end
    table_ind_max_sum = sum(table_ind_max,2);
    [~,ii] = max(table_ind_max_sum);
    [ind1, ind2, ind3, ind4, ind5, ind6]=ind2sub(size(F1mean),list_ind_max(ii));
    opt_rdmin(CV) = list_rdmin(ind1);
    opt_min_corr(CV) = list_min_corr(ind2);
    opt_min_pnr(CV) = list_min_pnr(ind3);
    opt_merge_thr(CV) = list_merge_thr(ind4);
    opt_merge_thr_spatial(CV) = list_merge_thr_spatial(ind5);
    opt_merge_thr_temporal(CV) = list_merge_thr_temporal(ind6);
    list_F1_CV(CV) = list_F1(CV, ind1, ind2, ind3, ind4, ind5, ind6);
    list_Recall_CV(CV) = list_Recall(CV, ind1, ind2, ind3, ind4, ind5, ind6);
    list_Precision_CV(CV) = list_Precision(CV, ind1, ind2, ind3, ind4, ind5, ind6);
    list_time_CV(CV) = list_time(CV, ind1, ind2, ind3, ind4, ind5, ind6);
%     list_F1_default(CV) = list_F1(CV, ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
%     list_Recall_default(CV) = list_Recall(CV, ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
%     list_Precision_default(CV) = list_Precision(CV, ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
end

Table_time = [opt_rdmin, opt_min_corr, opt_min_pnr, ...
    opt_merge_thr, opt_merge_thr_spatial, opt_merge_thr_temporal, ...
    list_Recall_max, list_Precision_max, list_F1_max, list_time_CV]; % , ...
%     list_Recall_CV, list_Precision_CV, list_F1_CV, ...
%     list_Recall_default, list_Precision_default, list_F1_default];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
gcf;

%% Show optimal parameters from F1All (CV).mat for CaImAn dataset
% load('eval_1p_small - Copy.mat');
% load('eval_1p_small.mat');

% min_corr_default = 0.8;
% min_pnr_default = 8;
% merge_thr_default = 0.6; % 0.65;
% merge_thr_spatial_default = 0.8;
% merge_thr_temporal_default = 0.4;
% nk_default = 3;
% ind1_default = find(list_nk == nk_default);
% ind2_default = find(list_min_corr == min_corr_default);
% ind3_default = find(list_min_pnr == min_pnr_default);
% ind4_default = find(list_merge_thr == merge_thr_default);
% ind5_default = find(list_merge_thr_spatial == merge_thr_spatial_default);
% ind6_default = find(list_merge_thr_temporal == merge_thr_temporal_default);
 % [list_F1_max,list_Recall_max,list_Precision_max,list_cnn_lowest,list_cnn_thr,list_min_SNR,list_rval_thr]=deal(zeros(1,4));

figure('Position',[0,100,1920,500]);

shape_F1 = size(list_F1);
F1mean = reshape(mean(list_F1(1:9,:,:,:,:,:,:),1),shape_F1(2:end));
Recallmean = reshape(mean(list_Recall(1,:,:,:,:,:,:),1),shape_F1(2:end));
Precisionmean = reshape(mean(list_Precision(1,:,:,:,:,:,:),1),shape_F1(2:end));
[F1_max, ~]=max(F1mean(:));
list_ind_max = find(F1mean==F1_max);
length_max = length(list_ind_max);
table_ind_max = zeros(length_max,6);
for ii = 1:length_max
    [ind1, ind2, ind3, ind4, ind5, ind6]=ind2sub(size(F1mean),list_ind_max(ii));
    table_ind_max(ii,:) = [ind1, ind2, ind3, ind4, ind5, ind6];
end
table_ind_max_sum = sum(table_ind_max,2);
[~,ii] = max(table_ind_max_sum);
[ind1, ind2, ind3, ind4, ind5, ind6]=ind2sub(size(F1mean),list_ind_max(ii));
nk = list_nk(ind1);
min_corr = list_min_corr(ind2);
min_pnr = list_min_pnr(ind3);
merge_thr = list_merge_thr(ind4);
merge_thr_spatial = list_merge_thr_spatial(ind5);
merge_thr_temporal = list_merge_thr_temporal(ind6);
% F1_max = F1_max;
Recall_max = Recallmean(list_ind_max(ii));
Precision_max = Precisionmean(list_ind_max(ii));
% F1_default = F1mean(ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
% Recall_default = Recallmean(ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);
% Precision_default = Precisionmean(ind1_default,ind2_default,ind3_default,ind4_default,ind5_default,ind6_default);

% subplot(2,3,1);    hold on;
% plot(list_nk,squeeze(F1mean(:,ind2,ind3,ind4,ind5,ind6)),'LineWidth',2);
% plot(list_nk(squeeze(F1mean(:,ind2,ind3,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(nk_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,1);    hold on;
plot(list_min_corr,squeeze(F1mean(ind1,:,ind3,ind4,ind5,ind6)),'LineWidth',2);
plot(list_min_corr(squeeze(F1mean(ind1,:,ind3,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(min_corr_default,F1_default,'*','Color',green,'LineWidth',2);
% subplot(2,3,3);    hold on;
% plot(list_min_pnr,squeeze(F1mean(ind1,ind2,:,ind4,ind5,ind6)),'LineWidth',2);
% plot(list_min_pnr(squeeze(F1mean(ind1,ind2,:,ind4,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(min_pnr_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,2);    hold on;
plot(list_merge_thr,squeeze(F1mean(ind1,ind2,ind3,:,ind5,ind6)),'LineWidth',2);
plot(list_merge_thr(squeeze(F1mean(ind1,ind2,ind3,:,ind5,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_thr_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,3);    hold on;
plot(list_merge_thr_spatial,squeeze(F1mean(ind1,ind2,ind3,ind4,:,ind6)),'LineWidth',2);
plot(list_merge_thr_spatial(squeeze(F1mean(ind1,ind2,ind3,ind4,:,ind6))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_thr_spatial_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,4);    hold on;
plot(list_merge_thr_temporal,squeeze(F1mean(ind1,ind2,ind3,ind4,ind5,:)),'LineWidth',2);
plot(list_merge_thr_temporal(squeeze(F1mean(ind1,ind2,ind3,ind4,ind5,:))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_thr_temporal_default,F1_default,'*','Color',green,'LineWidth',2);

% disp(['F1_max = ',replace(num2str(list_F1_max),'     ',', ')])
% disp(['rval_thr = ',replace(mat2str(list_rval_thr),' ',', ')])
% disp(['min_SNR = ',replace(mat2str(list_min_SNR),' ',', ')])
% disp(['cnn_thr = ',replace(mat2str(list_cnn_thr),' ',', ')])
% disp(['cnn_lowest = ',replace(mat2str(list_cnn_lowest),' ',', ')])
% Table=[Recall_max; Precision_max; F1_max; ...
%     pix_select_sigthres; pix_select_corrthres; merge_roi_corrthres; th_binary]';
% Table_ext = [Table; mean(Table,1); std(Table,1,1)];

clear ax
% ax(1)=subplot(1,4,1);
% % set(gca,'Fontsize',12);
% % expID = cellfun(@num2str, (num2cell(0:3)), 'UniformOutput',false);
% % legend(expID,'Location','South','NumColumns',2)
% xlabel('nk','Interpreter','None');
ax(1)=subplot(1,4,1);
% set(gca,'Fontsize',12);
ylabel('{\itF_1}');
xlabel('min_corr','Interpreter','None');
% ax(3)=subplot(1,4,3);
% % set(gca,'Fontsize',12);
% xlabel('min_pnr','Interpreter','None');
ax(2)=subplot(1,4,2);
% set(gca,'Fontsize',12);
xlabel('merge_thr','Interpreter','None');
ax(3)=subplot(1,4,3);
% set(gca,'Fontsize',12);
xlabel('merge_thr_spatial','Interpreter','None');
ax(4)=subplot(1,4,4);
% set(gca,'Fontsize',12);
xlabel('merge_thr_temporal','Interpreter','None');
linkaxes(ax,'y');
set(ax,'Fontsize',12);
% ylim([0.72,0.77]);
% suptitle('CaImAn batch parameter dependence')

% saveas(gcf,'CNMFE_mean_F1 vs params 1p_small 1x3x1x3x3x3 ring1.5.png')

%%
load('eval_1p_small.mat');
v1 = load('eval_1p_small - Copy.mat');
v2 = load('eval_1p_small - Copy (2).mat');
list_time_old = v2.list_time;
list_time_old(:,1,:,2,:,:,:) = max(list_time_old(:,1,:,2,:,:,:),v1.list_time(:,1,:,2,:,:,:));
list_time = max(list_time, list_time_old);
save('eval_1p_small_1x3x3x3x3x3_exp1-3.mat','list_Recall','list_Precision','list_F1','list_time',...
        'list_nk','list_merge_thr','list_merge_thr_spatial','list_merge_thr_temporal','list_min_corr','list_min_pnr')

%%
load('E:\simulation_CNMFE_corr_noise\lowBG=1e+03,poisson=0.1\CNMFE\gSiz=12,rbg=1.8,nk=1,rdmin=3.0,mc=0.20,mp=2,mt=0.20,mts=0.80,mtt=0.40\sim_0_result.mat','neuron')
Y = h5read('E:\simulation_CNMFE_corr_noise\lowBG=1e+03,poisson=0.1\sim_0.h5','/mov');
A3 = neuron.reshape(neuron.A, 2);
C = neuron.C;
A3s = imresize(A3,1/2);
Y3s = imresize(Y,1/2);
A2s = reshape(A3s,[],size(A3s,3));
Y2s = reshape(Y3s,[],size(Y3s,3));
Bf = Y2s - A2s * C;
Bf0 = Bf - mean(Bf,2);
W = neuron.W{1};
WBf0 = full(W)*Bf0;
Bf03 = reshape(Bf0,size(Y3s));
WBf03 = reshape(WBf0,size(Y3s));

figure; imagesc(Bf0); colorbar;
figure; imagesc(WBf0); colorbar;
figure; imagesc(WBf0 - Bf0); colorbar;
figure; imshow3D(Bf03);
figure; imshow3D(WBf03);

%%
Y = h5read('E:\simulation_CNMFEBG_noise\amp=0.01,poisson=1\sim_0.h5','/mov');
Y_SNR = h5read('E:\simulation_CNMFEBG_noise\amp=0.01,poisson=1\complete_TUnCaT\network_input\sim_0.h5','/network_input');
load('E:\simulation_CNMFEBG_noise\amp=0.01,poisson=1\GT info\GT_sim_0.mat')
sn = reshape(sn,64,64);
Y_b0 = Y - sn;
figure; imshow3D(Y);
figure; imshow3D(Y_b0,[-200,200]);
figure; imshow3D(Y_SNR,[-2,8]);
