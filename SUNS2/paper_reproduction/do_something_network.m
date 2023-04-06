addpath('C:\Matlab Files\timer');
timer_start_next;
% try
% CNMFE_par_data_TENASPIS_direvl_thb;
% end
try
% min1pipe_par_simu_direvl;
min1pipe_par_data_TENASPIS_direvl;
% CNMFE_simu_init_for_par;
% CNMFE_par_simu_direvl_thb;
% CNMFE_par_data_CNMFE_add;
% Copy_of_CNMFE_par_data_CNMFE_add;
% Copy_2_of_CNMFE_par_data_CNMFE_add;
% Copy_3_of_CNMFE_par_data_CNMFE_add;
% CNMFE_par_data_CNMFE_direvl_thb;
end
timer_stop;

%%
dir_parent='E:\data_CNMFE\';
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for vid= 2 % 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{vid};
    fname=fullfile(dir_parent,[Exp_ID,'.mat']);
    load(fname, 'Y');
    type = class(Y);
    h5_name = fullfile(dir_parent,[Exp_ID,'.h5']);
    if exist(h5_name,'file')
        delete(h5_name)
    end
    h5create(h5_name,'/mov',size(Y),'Datatype',type);
    h5write(h5_name,'/mov',Y);
end
%%
part = 'part11';
load(['E:\data_CNMFE\PFC4_15Hz_updated_masks_auto\complete_TUnCaT\4816[1]th5\output_masks\Output_Masks_PFC4_15Hz_',part,'.mat'])
FinalMasks = permute(Masks,[3,2,1]);
plot_masks_id_color(FinalMasks);
title(part);

%%
select_frames_sort = select_frames(select_frames_order);
weight_trace = list_weight_trace{nn}(select_frames_sort);
weight_frame = list_weight_frame{nn}(select_frames_sort);
weight = list_weight{nn}(select_frames_sort);
temp = [select_frames_sort;weight_trace;weight_frame;weight];
%%
max_in = max(avg_frame(mask_sub),[],'all');
figure; imshow3D(video_sub(:,:,select_frames_sort),[0,max_in]);
%%
figure;
ns = 1;
frame = video_sub(:,:,5023);
% frame = video_sub(:,:,select_frames_sort(ns));
imagesc(frame);
% imagesc(avg_frame_use);
colormap gray;
axis('image');
hold on;
contour(mask_sub,'b');
% contour(frame > max_in);
%%
sort_inten1 = sort_inside(:,select_frames_sort(ns));
sort_inten2 = sort_outside(:,select_frames_sort(ns));
sort_inten = [sort_inten1,sort_inten2(1:length(sort_inten1))]';

%% Check Preprocessing
nframes=1000;
start=[1,1,7000];
count=[Inf,Inf,nframes];
stride=[1,1,1];
network_input = h5read('D:\ABO\20 percent\ShallowUNet\network_input\539670003.h5','/network_input',start, count, stride);
temporal_masks = h5read('D:\ABO\20 percent\ShallowUNet\temporal_masks(6)\539670003.h5','/temporal_masks',start, count, stride);
start=[1,start];
count=[1,count];
stride=[1,stride];
pmaps = squeeze(h5read('D:\ABO\20 percent\ShallowUNet\probability_map\539670003.h5','/probability_map',start, count, stride));
combine = cat(2, temporal_masks*255, network_input, pmaps);
figure; imshow3D(combine);


%% Plot training loss
% file_training_output='training_output.h5';
dir_parent='E:\data_CNMFE\PFC4_15Hz\';
% list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
% dir_sub='complete-FISSA\test_CNN\4816[1]th4'; %_subtract
dir_sub='noSF_TUnCaT\4816[1]th4'; %_subtract
dir_training_output=[dir_parent,dir_sub,'\training output\training_output_CV'];

% list_Exp_ID = {'c25_1NRE','c27_NN','c28_1NRE1NLE'};
% dir_video = 'E:\OnePhoton videos\full videos\';
% dir_sub='complete\test_CNN\4816[1,2]th4'; %_subtract
% dir_training_output=[dir_video,dir_sub,'\training output\training_output_CV'];

num_Exp = 4; % length(list_Exp_ID);
[list_loss,list_bce_loss,list_dice_loss,list_val_loss,list_val_bce_loss,...
    list_val_dice_loss] = deal(zeros(num_Exp,1));
for CV=0:num_Exp-1
    file_training_output=[dir_training_output,num2str(CV),'.h5'];
    val_loss=h5read(file_training_output,'/val_loss');
    val_dice_loss=h5read(file_training_output,'/val_dice_loss');
    loss=h5read(file_training_output,'/loss');
    dice_loss=h5read(file_training_output,'/dice_loss');
%     range=50;
%     val_loss=val_loss(1:range);
%     val_dice_loss=val_dice_loss(1:range);
%     loss=loss(1:range);
%     dice_loss=dice_loss(1:range);
    val_bce_loss=val_loss-val_dice_loss;
    bce_loss=loss-dice_loss;
    list_loss(CV+1) = loss(end);
    list_bce_loss(CV+1) = bce_loss(end);
    list_dice_loss(CV+1) = dice_loss(end);
    list_val_loss(CV+1) = val_loss(end);
    list_val_bce_loss(CV+1) = val_bce_loss(end);
    list_val_dice_loss(CV+1) = val_dice_loss(end);
    %%
    figure; 
    set(gcf, 'Position', [100,100,1300,400]);
    subplot(1,3,1)
    plot([loss,val_loss],'LineWidth',2);
    ylabel('Loss');
    xlabel('Epoch');
    % title(Exp_ID);
    title(['Cross Validation ', num2str(CV)]);
    legend('Training loss','Validation loss');
    ylim([0,1])

    subplot(1,3,2)
    plot([dice_loss,val_dice_loss],'LineWidth',2);
    ylabel('Loss');
    xlabel('Epoch');
    % title(Exp_ID);
    title(['Cross Validation ', num2str(CV)]);
    legend('Training dice loss','Validation dice loss');
    ylim([0,1])
    
    subplot(1,3,3)
    plot([bce_loss,val_bce_loss],'LineWidth',2);
    ylabel('Loss');
    xlabel('Epoch');
    % title(Exp_ID);
    title(['Cross Validation ', num2str(CV)]);
    legend('Training bce loss','Validation bce loss');
    
%     figure; 
%     set(gcf, 'Position', [100,100,900,400]);
%     subplot(1,2,1)
%     plot([loss,dice_loss,bce_loss],'LineWidth',2);
%     ylabel('Loss');
%     xlabel('Epoch');
%     % title(Exp_ID);
%     title(['Cross Validation ', num2str(CV)]);
%     legend('Training loss','Training dice loss','Training bce loss');
%     ylim([0,1])
% 
%     subplot(1,2,2)
%     plot([val_loss,val_dice_loss,val_bce_loss],'LineWidth',2);
%     ylabel('Loss');
%     xlabel('Epoch');
%     % title(Exp_ID);
%     title(['Cross Validation ', num2str(CV)]);
%     legend('Validation loss','Validation dice loss','Validation bce loss');
    ylim([0,2])

    saveas(gcf,[dir_training_output,num2str(CV),'.png']);
end
loss_all = [list_loss,list_bce_loss,list_dice_loss,list_val_loss,list_val_bce_loss,list_val_dice_loss];
loss_all_ext = [loss_all; nanmean(loss_all,1); nanstd(loss_all,1,1)];
nanmean(loss_all,1)

%% Plot training loss vs th_SNR
% file_training_output='training_output.h5';
dir_parent = 'E:\1photon-small\';
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
num_Exp = length(list_Exp_ID);
list_thred_ratio = 4:10; % 3:6; % 
num_thred_ratio = length(list_thred_ratio);
[list_loss,list_bce_loss,list_dice_loss,list_val_loss,list_val_bce_loss,...
    list_val_dice_loss] = deal(zeros(num_Exp,num_thred_ratio));

for thid = 1:num_thred_ratio
    thred_ratio = list_thred_ratio(thid);
%     dir_sub=['complete-FISSA\test_CNN\4816[1]th',num2str(thred_ratio)]; %_subtract
    dir_sub=['complete\test_CNN\4816[1]th',num2str(thred_ratio),'+BGlayer']; %_subtract
    dir_training_output=[dir_parent,dir_sub,'\training output\training_output_CV'];

    % list_Exp_ID = {'c25_1NRE','c27_NN','c28_1NRE1NLE'};
    % dir_video = 'E:\OnePhoton videos\full videos\';
    % dir_sub='complete\test_CNN\4816[1,2]th4'; %_subtract
    % dir_training_output=[dir_video,dir_sub,'\training output\training_output_CV'];

    for CV=0:num_Exp-1
        file_training_output=[dir_training_output,num2str(CV),'.h5'];
        val_loss=h5read(file_training_output,'/val_loss');
        val_dice_loss=h5read(file_training_output,'/val_dice_loss');
        loss=h5read(file_training_output,'/loss');
        dice_loss=h5read(file_training_output,'/dice_loss');
    %     range=50;
    %     val_loss=val_loss(1:range);
    %     val_dice_loss=val_dice_loss(1:range);
    %     loss=loss(1:range);
    %     dice_loss=dice_loss(1:range);
        val_bce_loss=val_loss-val_dice_loss;
        bce_loss=loss-dice_loss;
        list_loss(CV+1,thid) = loss(end);
        list_bce_loss(CV+1,thid) = bce_loss(end);
        list_dice_loss(CV+1,thid) = dice_loss(end);
        list_val_loss(CV+1,thid) = val_loss(end);
        list_val_bce_loss(CV+1,thid) = val_bce_loss(end);
        list_val_dice_loss(CV+1,thid) = val_dice_loss(end);
    end
end
loss_all = cat(3,list_loss,list_bce_loss,list_dice_loss,list_val_loss,list_val_bce_loss,list_val_dice_loss);
loss_all_ext = [loss_all; nanmean(loss_all,1); nanstd(loss_all,1,1)];
loss_all_mean = squeeze(nanmean(loss_all,1));
loss_all_mean_T = loss_all_mean';

%% Generate a table of optimal parameters and F1 for 1 photon. Train and test on different videos
% clear
% dir_parent='E:\simulation_CNMFE_randBG\noise30\'; % , PFC4_15Hz
% dir_parent='E:\simulation_CNMFE_corr_noise\lowBG=5e+03,poisson=0.3\'; % , PFC4_15Hz
% dir_parent='E:\simulation_constantBG_noise\amp=0.003,poisson=1\'; % , PFC4_15Hz
% num_Exp = 9;
% dir_parent='E:\data_CNMFE\blood_vessel_10Hz_original_masks\'; % 
% dir_parent='E:\data_CNMFE\PFC4_15Hz_original_masks\'; % 
% dir_parent='E:\data_CNMFE\PFC4_15Hz_added_blockwise_weighted_sum_unmask\'; % 
% num_Exp = 4;
dir_parent='D:\data_TENASPIS\added_refined_masks\'; % 
num_Exp = 8;
sub_add='add_neurons_0.02_rotate'; % ''; % 
dir_sub='\complete_TUnCaT_SF25_train_from_real_fake\4816[1]th5'; %_lowoverlap
dir_sub_train='\complete_TUnCaT_SF25\4816[1]th5'; %
% dir_sub='\complete_FISSA\4816[1]th3'; %
dir_output_masks=fullfile(dir_parent,dir_sub,['output_masks ',sub_add]); % online
dir_params=fullfile(dir_parent,sub_add,dir_sub_train,'output_masks');
dir_output_info=fullfile(dir_output_masks,'Output_Info_');
dir_optim_info=fullfile(dir_params,'Optimization_Info_');
% list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%     'c25_163_267','c27_114_176','c28_161_149',...
%     'c25_123_348','c27_122_121','c28_163_244'};
[F1_all, Recall_all, Precision_all, Time_frame_all, ...
    minArea_all, avgArea_all, thresh_pmap_all, thresh_COM_all, ...
    thresh_IOU_all, thresh_consume_all, cons_all, win_avg_all] = deal(zeros(num_Exp,1));
Table_max = zeros(num_Exp,3);

OutputInfo = load([dir_output_info,'All.mat']);
F1_all=OutputInfo.list_F1;
Recall_all=OutputInfo.list_Recall;
Precision_all=OutputInfo.list_Precision;
list_time = OutputInfo.list_time;
for CV=1:num_Exp %[1,2,4:10] %
%     Exp_ID=list_Exp_ID{CV};
%     Params=OutputInfo.Params;

%     OptimInfo = load([dir_optim_info,Exp_ID,'.mat']);
    load([dir_optim_info,num2str(CV-1),'.mat'], 'Params','Table'); %OptimInfo = 
%     load([dir_optim_info,num2str(10),'.mat'], 'Params','Table'); %OptimInfo = 
%     Params=OptimInfo.Params;
    minArea_all(CV)=Params.minArea;
    avgArea_all(CV)=Params.avgArea;
    thresh_pmap_all(CV)=Params.thresh_pmap;
    thresh_COM_all(CV)=Params.thresh_COM;
    thresh_IOU_all(CV)=Params.thresh_IOU;
    thresh_consume_all(CV)=Params.thresh_consume;
    cons_all(CV)=Params.cons;
%     win_avg_all(CV)=Params.win_avg;    
%     Time_frame_all(CV)=OutputInfo.Time_frame;
%     Table = OptimInfo.Table;
    [~, indmax] = max(Table(:,end));
    Table_max(CV,:) = Table(indmax,end-2:end);
end
    
Params_all=[minArea_all, avgArea_all, thresh_pmap_all, thresh_COM_all, ...
    cons_all, Recall_all, Precision_all, F1_all, Table_max, list_time(:,end)]; %, Time_frame_all
% win_avg_all, thresh_IOU_all, thresh_consume_all, 
Params_all_ext=[Params_all;nanmean(Params_all,1);nanstd(Params_all,1,1)]; %([1,2,4:10],:)
disp([mean(Recall_all),mean(Precision_all),mean(F1_all)])


%% Generate a table of optimal parameters and F1 for 1 photon
% clear
% dir_parent='E:\simulation_CNMFE_randBG\noise30\'; % , PFC4_15Hz
% dir_parent='E:\simulation_CNMFE_corr_noise\lowBG=5e+03,poisson=0.3\'; % , PFC4_15Hz
% dir_parent='E:\simulation_constantBG_noise\amp=0.003,poisson=1\'; % , PFC4_15Hz
% num_Exp = 9;
% dir_parent='E:\data_CNMFE\blood_vessel_10Hz\'; % 
% dir_parent='E:\data_CNMFE\PFC4_15Hz\'; % 
% dir_parent='E:\data_CNMFE\bma22_epm\'; % 
% dir_parent='E:\data_CNMFE\CaMKII_120_TMT Exposure_5fps\'; % 
% num_Exp = 4;
dir_parent='D:\data_TENASPIS\added_refined_masks\'; % add_neurons_0.003_rotate\
num_Exp = 8;
dir_sub='\complete_TUnCaT_SF25\4816[1]th5'; %_2out+BGlayer
% dir_sub='\complete_FISSA_SF25\4816[1]th2'; %_2out+BGlayer
% dir_sub='\complete_FISSA_noSF\4816[1]th3'; %_2out+BGlayer
dir_output_masks=fullfile(dir_parent,dir_sub,'output_masks_CPU'); % online
dir_params=fullfile(dir_parent,dir_sub,'output_masks');
dir_output_info=fullfile(dir_output_masks,'Output_Info_');
dir_optim_info=fullfile(dir_params,'Optimization_Info_');
% list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
%     'c25_163_267','c27_114_176','c28_161_149',...
%     'c25_123_348','c27_122_121','c28_163_244'};
[F1_all, Recall_all, Precision_all, Time_frame_all, ...
    minArea_all, avgArea_all, thresh_pmap_all, thresh_COM_all, ...
    thresh_IOU_all, thresh_consume_all, cons_all, win_avg_all] = deal(zeros(num_Exp,1));
Table_max = zeros(num_Exp,3);

OutputInfo = load([dir_output_info,'All.mat']);
F1_all=OutputInfo.list_F1;
Recall_all=OutputInfo.list_Recall;
Precision_all=OutputInfo.list_Precision;
list_time = OutputInfo.list_time;
for CV=1:num_Exp %[1,2,4:10] %
%     Exp_ID=list_Exp_ID{CV};
%     Params=OutputInfo.Params;

%     OptimInfo = load([dir_optim_info,Exp_ID,'.mat']);
    load([dir_optim_info,num2str(CV-1),'.mat'], 'Params','Table'); %OptimInfo = 
%     load([dir_optim_info,num2str(10),'.mat'], 'Params','Table'); %OptimInfo = 
%     Params=OptimInfo.Params;
    minArea_all(CV)=Params.minArea;
    avgArea_all(CV)=Params.avgArea;
    thresh_pmap_all(CV)=Params.thresh_pmap;
    thresh_COM_all(CV)=Params.thresh_COM;
    thresh_IOU_all(CV)=Params.thresh_IOU;
    thresh_consume_all(CV)=Params.thresh_consume;
    cons_all(CV)=Params.cons;
%     win_avg_all(CV)=Params.win_avg;    
%     Time_frame_all(CV)=OutputInfo.Time_frame;
%     Table = OptimInfo.Table;
    [~, indmax] = max(Table(:,end));
    Table_max(CV,:) = Table(indmax,end-2:end);
end
    
Params_all=[minArea_all, avgArea_all, thresh_pmap_all, thresh_COM_all, ...
    cons_all, Recall_all, Precision_all, F1_all, Table_max, list_time(:,end)]; %, Time_frame_all
% win_avg_all, thresh_IOU_all, thresh_consume_all, 
Params_all_ext=[Params_all;nanmean(Params_all,1);nanstd(Params_all,1,1)]; %([1,2,4:10],:)
disp([mean(Recall_all),mean(Precision_all),mean(F1_all)])

%% Generate a table of recall, precision and F1 for 1 photon after missing finder
dir_parent='D:\data_TENASPIS\added_refined_masks\';
dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT_SF25\4816[1]th5'); % 4 v1
dir_masks = fullfile(dir_SUNS, 'output_masks');
sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
dir_add_new = fullfile(dir_masks, sub_folder);

d0 = 0.8;
lam = 20; % [10,15,20] % [5,8,10] % 
optimize_th_cnn = false; % true; % 
res =  0; % [0,1,3:9]; %
num_frame = 0; % [0,2:8]; % 
mask_option = 'nomask'; % {'nomask', 'mask', 'bmask', 'Xmask'};
num_mask_channel = 1; % [1,2]; % 
shuffle = ''; % '_shuffle'; % 
if num_frame <= 1
    img_option = 'avg';
    str_shuffle = '';
else
    img_option = 'multi_frame';
    str_shuffle = shuffle;
end
sub1 = sprintf('%s_%s',img_option,mask_option);
if optimize_th_cnn
    sub1 = [sub1,'_0.5'];
end
sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
sub0 = sprintf('trained dropout %gexp(-%g)',d0,lam);
folder = fullfile(sub0,sub1,sub2);

load(fullfile(dir_add_new,folder,'eval.mat'),'list_Recall_add',...
    'list_Recall','list_Precision','list_F1','list_time');
% Table = [list_Recall_add,list_Recall,list_Precision,list_F1,list_time(:,end)];
Table = [list_Recall,list_Precision,list_F1];
Table_ext = [Table;nanmean(Table,1);nanstd(Table,1,1)];
used_time = list_time(:,end);
used_time_ext = [used_time;nanmean(used_time,1);nanstd(used_time,1,1)];

%%
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
path_name = 'D:\data_TENASPIS\added_refined_masks';
dir_valid = 'D:\data_TENASPIS\original_masks\GT Masks\add_new_blockwise_weighted_sum_unmask';
% dir_valid = fullfile(path_name,'GT Masks dropout 0.8exp(-15)\add_new_blockwise_weighted_sum_unmask');
% dir_valid = fullfile(path_name,'complete_TUnCaT_SF25\4816[1]th5\output_masks\add_new_blockwise_weighted_sum_unmask\trained dropout 0.8exp(-15)\avg_Xmask_0.5\classifier_res0_0+1 frames');
list_result = cell(1,num_Exp);
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_valid,[Exp_ID,'_added_CNNtrain_blockwise.mat']),'list_valid');
    list_result{eid} = list_valid;
%     load(fullfile(dir_valid,['CNN_predict_',Exp_ID,'_cv',num2str(eid-1),'.mat']),'pred_valid');
%     list_result{eid} = pred_valid;
end
list_result_mat = cell2mat(list_result);
mean(list_result_mat)
