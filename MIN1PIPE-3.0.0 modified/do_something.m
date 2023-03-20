addpath('C:\Matlab Files\timer');
timer_start_next;
try
% min1pipe_simu_direvl;
% min1pipe_par_data_CNMFE_direvl;
% min1pipe_par_data_TENASPIS_cv;
% min1pipe_par_data_TENASPIS_direvl;
% min1pipe_data_TENASPIS_direvl_test;
% min1pipe_data_CNMFE_direvl_cv_test;
% min1pipe_data_TENASPIS_direvl_cv_test;
min1pipe_par_simu_direvl;
% min1pipe_par_simu_direvl_cv;
% min1pipe_simu_direvl_cv_test;
end
timer_stop;

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

%% Show optimal parameters for max mean_F1
% load('eval_1p_small_vary_e19 (2).mat');

pix_select_sigthres_default = 0.8;
pix_select_corrthres_default = 0.6;
merge_roi_corrthres_default = 0.9;
th_binary_default = 0.2;
ind1_default = find(list_pss == pix_select_sigthres_default);
ind2_default = find(list_psc == pix_select_corrthres_default);
ind3_default = find(list_mrc == merge_roi_corrthres_default);
ind4_default = find(list_thb == th_binary_default);
 % [list_F1_max,list_Recall_max,list_Precision_max,list_cnn_lowest,list_cnn_thr,list_min_SNR,list_rval_thr]=deal(zeros(1,4));

figure('Position',[0,100,1920,500]);

shape_F1 = size(list_F1);
F1mean = reshape(mean(list_F1,1),shape_F1(2:end));
Recallmean = reshape(mean(list_Recall,1),shape_F1(2:end));
Precisionmean = reshape(mean(list_Precision,1),shape_F1(2:end));
[F1_max, ind_max]=nanmax(F1mean(:));
list_ind_max = find(F1mean==F1_max);
length_max = length(list_ind_max);
table_ind_max = zeros(length_max,4);
for ii = 1:length_max
    [ind1, ind2, ind3, ind4]=ind2sub(size(F1mean),list_ind_max(ii));
    table_ind_max(ii,:) = [ind1, ind2, ind3, ind4];
end
table_ind_max_sum = sum(table_ind_max,2);
[~,ii] = max(table_ind_max_sum);
[ind1, ind2, ind3, ind4]=ind2sub(size(F1mean),list_ind_max(ii));
pix_select_sigthres = list_pss(ind1);
pix_select_corrthres = list_psc(ind2);
merge_roi_corrthres = list_mrc(ind3);
th_binary = list_thb(ind4);
% F1_max = F1_max;
Recall_max = Recallmean(ind_max);
Precision_max = Precisionmean(ind_max);
list_time_opt = squeeze(list_time(:, ind1, ind2, ind3, ind4));
list_Recall_max = squeeze(list_Recall(:, ind1, ind2, ind3, ind4));
list_Precision_max = squeeze(list_Precision(:, ind1, ind2, ind3, ind4));
list_F1_max = squeeze(list_F1(:, ind1, ind2, ind3, ind4));
F1_default = F1mean(ind1_default,ind2_default,ind3_default,ind4_default);
Recall_default = Recallmean(ind1_default,ind2_default,ind3_default,ind4_default);
Precision_default = Precisionmean(ind1_default,ind2_default,ind3_default,ind4_default);

subplot(1,4,1);    hold on;
plot(list_pss,squeeze(F1mean(:,ind2,ind3,ind4)),'LineWidth',2);
plot(list_pss(squeeze(F1mean(:,ind2,ind3,ind4))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(pix_select_sigthres_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,2);    hold on;
plot(list_psc,squeeze(F1mean(ind1,:,ind3,ind4)),'LineWidth',2);
plot(list_psc(squeeze(F1mean(ind1,:,ind3,ind4))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(pix_select_corrthres_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,3);    hold on;
plot(list_mrc,squeeze(F1mean(ind1,ind2,:,ind4)),'LineWidth',2);
plot(list_mrc(squeeze(F1mean(ind1,ind2,:,ind4))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(merge_roi_corrthres_default,F1_default,'*','Color',green,'LineWidth',2);
subplot(1,4,4);    hold on;
plot(list_thb,squeeze(F1mean(ind1,ind2,ind3,:)),'LineWidth',2);
plot(list_thb(squeeze(F1mean(ind1,ind2,ind3,:))==F1_max),F1_max,'*','Color',red,'LineWidth',2);
% plot(th_binary_default,F1_default,'*','Color',green,'LineWidth',2);

% Table=[Recall_max; Precision_max; F1_max; ...
%     pix_select_sigthres; pix_select_corrthres; merge_roi_corrthres; th_binary]';
% Table_ext = [Table; mean(Table,1); std(Table,1,1)];
Table_eval=[list_Recall_max, list_Precision_max, list_F1_max, list_time_opt]; 
Table_params = [pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, th_binary];
Table = [repmat(Table_params,[4,1]), Table_eval];
Table_ext = [Table; mean(Table,1); std(Table,1,1)];


ax(1)=subplot(1,4,1);
% set(gca,'Fontsize',12);
% expID = cellfun(@num2str, (num2cell(0:3)), 'UniformOutput',false);
% legend(expID,'Location','South','NumColumns',2)
ylabel('{\itF_1}');
xlabel('pix_select_sigthres','Interpreter','None');
ax(2)=subplot(1,4,2);
% set(gca,'Fontsize',12);
xlabel('pix_select_corrthres','Interpreter','None');
ax(3)=subplot(1,4,3);
% set(gca,'Fontsize',12);
xlabel('merge_roi_corrthres','Interpreter','None');
ax(4)=subplot(1,4,4);
% set(gca,'Fontsize',12);
xlabel('th_binary','Interpreter','None');
linkaxes(ax,'y');
set(ax,'Fontsize',12);
ylim([0.25,0.35]);
% suptitle('CaImAn batch parameter dependence')

% saveas(gcf,'MIN1PIPE_mean_F1 vs params 1p_small.png')


%% Show optimal parameters from cross validation
% load('eval_1p_small_vary_e19.mat');
% 
% pix_select_sigthres_default = 0.8;
% pix_select_corrthres_default = 0.6;
% merge_roi_corrthres_default = 0.9;
% th_binary_default = 0.5;
% ind1_default = find(list_pss == pix_select_sigthres_default);
% ind2_default = find(list_psc == pix_select_corrthres_default);
% ind3_default = find(list_mrc == merge_roi_corrthres_default);
% ind4_default = find(list_thb == th_binary_default);

shape_F1 = size(list_F1);
num_Exp = shape_F1(1);
[list_F1_CV,list_Recall_CV,list_Precision_CV,list_time_CV,list_F1_default,list_Recall_default,list_Precision_default,...
    opt_pix_select_sigthres,opt_pix_select_corrthres,opt_merge_roi_corrthres,opt_th_binary]=deal(zeros(num_Exp,1));

for CV = 1:num_Exp
    train = setdiff(1:num_Exp,CV);
    F1mean = reshape(mean(list_F1(train,:,:,:,:),1),shape_F1(2:end));
%     Recallmean = reshape(mean(list_Recall(train,:,:,:,:),1),shape_F1(2:end));
%     Precisionmean = reshape(mean(list_Precision(train,:,:,:,:),1),shape_F1(2:end));
    [F1_max, ~]=nanmax(F1mean(:));
    list_ind_max = find(F1mean==F1_max);
    length_max = length(list_ind_max);
    table_ind_max = zeros(length_max,length(shape_F1)-1);
    for ii = 1:length_max
        [ind1, ind2, ind3, ind4]=ind2sub(size(F1mean),list_ind_max(ii));
        table_ind_max(ii,:) = [ind1, ind2, ind3, ind4];
    end
    table_ind_max_sum = sum(table_ind_max,2);
    [~,ii] = max(table_ind_max_sum);
    [ind1, ind2, ind3, ind4]=ind2sub(size(F1mean),list_ind_max(ii));
    opt_pix_select_sigthres(CV) = list_pss(ind1);
    opt_pix_select_corrthres(CV) = list_psc(ind2);
    opt_merge_roi_corrthres(CV) = list_mrc(ind3);
    opt_th_binary(CV) = list_thb(ind4);
    list_F1_CV(CV) = list_F1(CV, ind1, ind2, ind3, ind4);
    list_Recall_CV(CV) = list_Recall(CV, ind1, ind2, ind3, ind4);
    list_Precision_CV(CV) = list_Precision(CV, ind1, ind2, ind3, ind4);
    list_time_CV(CV) = list_time(CV, ind1, ind2, ind3, ind4);
%     list_F1_default(CV) = list_F1(CV, ind1_default,ind2_default,ind3_default,ind4_default);
%     list_Recall_default(CV) = list_Recall(CV, ind1_default,ind2_default,ind3_default,ind4_default);
%     list_Precision_default(CV) = list_Precision(CV, ind1_default,ind2_default,ind3_default,ind4_default);
end

Table_time = [opt_pix_select_sigthres, opt_pix_select_corrthres, ...
    opt_merge_roi_corrthres, opt_th_binary, ...
    list_Recall_max, list_Precision_max, list_F1_max, list_time_CV]; % , ...
%     list_Recall_CV, list_Precision_CV, list_F1_CV, ...
%     list_Recall_default, list_Precision_default, list_F1_default];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
