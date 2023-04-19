%% train on full
doesplot = 0;
list_res = 0; % [0,1,3:9]; % 
list_nframes = [0,2:8]; % 0; % 
num_res = length(list_res);
num_nf = length(list_nframes);
nv = max(num_res,num_nf);
Table = zeros(nv,4);
dir_parent = 'E:\data_CNMFE\full videos\GT Masks\add_new_blockwise';
mask_option = 'nomask'; % 'nomask', 'mask', 'bmask', 'Xmask';
shuffle = ''; % '_shuffle'; % 
num_mask_channel = 1; % 2; % 
optimize_th_cnn = false; % true; % 
for rid=1:num_res
    res = list_res(rid);
for nid=1:num_nf
    num_frame = list_nframes(nid);
    vid = max(rid,nid);
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
%     version = ['classifier'];
%     version = ['classifier_res0_',num2str(list_version(vid)),'+1 frames'];
%     version = ['classifier_res',num2str(list_version(vid)),'_0+2 frames'];
%     load(fullfile(dir_parent,'classifiers_multiframes_mask_thb_shuffle',...
%         version,'training_output_PFC4_15Hz.mat']);
    load(fullfile(dir_parent,sub1,sub2,'training_output_PFC4_15Hz.mat'));
    if optimize_th_cnn
        Table(vid,:) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
    else
        Table(vid,:) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
    end
%     disp([list_version(v),Table{v}]);

    if doesplot
        figure; 
        plot(loss); 
        hold on; 
        plot(accuracy); 
        legend({'loss','accuracy'},'Location','Southwest');
        xlabel('Epoch');
        ylim([0,1]);
        set(gca,'FontSize',14);
    %     saveas(gcf,['loss ',version,' train.png']);
    end
end
end
% Table = cell2mat(Table');
if num_nf > num_res
    [list_nframes',Table]
elseif num_nf < num_res
    [list_res',Table]
end

%% TENASPIS different nframes or CNN architecture
% dir_parent='D:\data_TENASPIS\original_masks\';
dir_video='D:\data_TENASPIS\added_refined_masks\';
dir_parent = fullfile(dir_video,'GT Masks dropout 0.8exp(-15)\add_new_blockwise_weighted_sum_unmask'); % GT Masks\
nvideo = 8;
% list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
% data_ind = 2;
% data_name = list_data_names{data_ind};
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
% dir_parent = fullfile(dir_parent,'GT Masks\add_new_blockwise'); % _weighted_sum_expanded_edge_unmask
% nvideo = 4;
doesplot = 0;
list_res = [0,1,3:9]; % 0; % 
list_nframes = 0; % [0,2:8]; % 
num_res = length(list_res);
num_nf = length(list_nframes);
nv = max(num_res,num_nf);
Table = zeros(nv,4,nvideo);
mask_option = 'Xmask'; % 'nomask', 'mask', 'bmask', 'Xmask';
shuffle = ''; % '_shuffle'; % 
num_mask_channel = 2; % 1; % 
optimize_th_cnn = true; % false; % 
for rid=1:num_res
    res = list_res(rid);
for nid=1:num_nf
    num_frame = list_nframes(nid);
    vid = max(rid,nid);
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
%         load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_%s_cv%d.mat',data_name,cv-1)));
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_cv%d.mat',cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','Southwest');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
        %     saveas(gcf,['loss ',version,' train.png']);
        end
    end
end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3);
if num_nf > num_res
    [list_nframes',Table_mean]
elseif num_nf < num_res
    [list_res',Table_mean]
end

%% res0_avg
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
data_ind = 2;
data_name = list_data_names{data_ind};
dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
dir_parent = fullfile(dir_parent,'GT Masks\add_new_blockwise_weighted_sum_unmask');
doesplot = 0; % _weighted_sum_expanded_edge_unmask
optimize_th_cnn = false; % true; % 
res = 0; % [0,1,3:9]; % 
num_frame = 0; % [0,2:8]; % 
nvideo = 4;
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table = zeros(nv,4,nvideo);
for moid=1:length(list_mask_option)
    mask_option = list_mask_option{moid};
for mcid=1:length(list_num_mask_channel)
    num_mask_channel = list_num_mask_channel(mcid);
    vid = (moid-1)*num_mc+(mcid-1);
    if strcmp(mask_option,'nomask')
        if num_mask_channel == 1
            vid = vid+1;
        elseif num_mask_channel == 2
            continue;
        end
    end
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_%s_cv%d.mat',data_name,cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','Southwest');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
        %     saveas(gcf,['loss ',version,' train.png']);
        end
    end
end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3)
% if num_nf > num_res
%     [list_nframes',Table_mean]
% elseif num_nf < num_res
%     [list_res',Table_mean]
% end

%% res0_avg add
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
data_ind = 2;
data_name = list_data_names{data_ind};
dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise_weighted_sum_unmask']);
neuron_amp = 0.001; % [0.001, 0.002, 0.003, 0.005];
dir_video = fullfile(dir_parent, ['add_neurons_',num2str(neuron_amp),'_rotate']);
dir_parent = fullfile(dir_video,'add_new_blockwise_weighted_sum_unmask'); % GT Masks\
doesplot = 0; % _weighted_sum_expanded_edge_unmask
optimize_th_cnn = true; % false; % 
res = 0; % [0,1,3:9]; % 
num_frame = 0; % [0,2:8]; % 
nvideo = 4;
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table = zeros(nv,4,nvideo);
for moid=1:length(list_mask_option)
    mask_option = list_mask_option{moid};
for mcid=1:length(list_num_mask_channel)
    num_mask_channel = list_num_mask_channel(mcid);
    vid = (moid-1)*num_mc+(mcid-1);
    if strcmp(mask_option,'nomask')
        if num_mask_channel == 1
            vid = vid+1;
        elseif num_mask_channel == 2
            continue;
        end
    end
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_%s_cv%d.mat',data_name,cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','Southwest');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
        %     saveas(gcf,['loss ',version,' train.png']);
        end
    end
end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3)
% if num_nf > num_res
%     [list_nframes',Table_mean]
% elseif num_nf < num_res
%     [list_res',Table_mean]
% end


%% res0_avg drop
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
data_ind = 1;
data_name = list_data_names{data_ind};
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
dir_parent=fullfile('E:\data_CNMFE\',data_name);
nvideo = 4;
d0 = 0.8;
lam = 5; % [10,15,20] % [5,8,10] % 
dir_video = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
dir_parent = fullfile(dir_video,'add_new_blockwise_weighted_sum_unmask'); % GT Masks\
doesplot = 0; % _weighted_sum_expanded_edge_unmask
optimize_th_cnn = false; % true; % 
res = 0; % [0,1,3:9]; % 
num_frame = 0; % [0,2:8]; % 
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table = zeros(nv,4,nvideo);
for moid=1:length(list_mask_option)
    mask_option = list_mask_option{moid};
for mcid=1:length(list_num_mask_channel)
    num_mask_channel = list_num_mask_channel(mcid);
    vid = (moid-1)*num_mc+(mcid-1);
    if strcmp(mask_option,'nomask')
        if num_mask_channel == 1
            vid = vid+1;
        elseif num_mask_channel == 2
            continue;
        end
    end
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_%s_cv%d.mat',data_name,cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','Southwest');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
        %     saveas(gcf,['loss ',version,' train.png']);
        end
    end
end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3)
% if num_nf > num_res
%     [list_nframes',Table_mean]
% elseif num_nf < num_res
%     [list_res',Table_mean]
% end


%% res0_avg drop for TENASPIS data
dir_parent='D:\data_TENASPIS\original_masks\';
% dir_parent='D:\data_TENASPIS\added_refined_masks\';
nvideo = 8;
d0 = 0.8;
lam = 20; % [10,15,20] % [5,8,10] % 
dir_video = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
dir_parent = fullfile(dir_video,'add_new_blockwise_weighted_sum_unmask'); % GT Masks\
doesplot = 0; % _weighted_sum_expanded_edge_unmask
optimize_th_cnn = true; % false; % 
res = 0; % [0,1,3:9]; % 
num_frame = 0; % [0,2:8]; % 
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table = zeros(nv,4,nvideo);
for moid=1:length(list_mask_option)
    mask_option = list_mask_option{moid};
for mcid=1:length(list_num_mask_channel)
    num_mask_channel = list_num_mask_channel(mcid);
    vid = (moid-1)*num_mc+(mcid-1);
    if strcmp(mask_option,'nomask')
        if num_mask_channel == 1
            vid = vid+1;
        elseif num_mask_channel == 2
            continue;
        end
    end
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_cv%d.mat',cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot % && cv==1
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','Southwest');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
            saveas(gcf,['loss curves\TENASPIS loss ',sprintf('dropout %gexp(-%g)',d0,lam),'_',sub1,'_',sub2,' train cv',num2str(cv),'.png']);
        end
    end
end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3)
% if num_nf > num_res
%     [list_nframes',Table_mean]
% elseif num_nf < num_res
%     [list_res',Table_mean]
% end


%% res0_avg add for TENASPIS data
% dir_parent='D:\data_TENASPIS\original_masks\';
dir_parent='D:\data_TENASPIS\added_refined_masks\';
nvideo = 8;
neuron_amp = 0.003; 
dir_video = fullfile(dir_parent, ['add_neurons_',num2str(neuron_amp),'_rotate']);
% dir_video = fullfile(dir_parent, 'GT Masks');
dir_parent = fullfile(dir_video,'add_new_blockwise_weighted_sum_unmask'); % GT Masks\
doesplot = 0; % _weighted_sum_expanded_edge_unmask
optimize_th_cnn = true; % false; % 
res = 0; % [0,1,3:9]; % 
num_frame = 0; % [0,2:8]; % 
list_mask_option = {'nomask', 'mask', 'bmask', 'Xmask'};
shuffle = ''; % '_shuffle'; % 
list_num_mask_channel = [1,2]; % 
num_mo = length(list_mask_option);
num_mc = length(list_num_mask_channel);
nv = num_mo*num_mc-1;
Table = zeros(nv,4,nvideo);
for moid=1:length(list_mask_option)
    mask_option = list_mask_option{moid};
for mcid=1:length(list_num_mask_channel)
    num_mask_channel = list_num_mask_channel(mcid);
    vid = (moid-1)*num_mc+(mcid-1);
    if strcmp(mask_option,'nomask')
        if num_mask_channel == 1
            vid = vid+1;
        elseif num_mask_channel == 2
            continue;
        end
    end
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_cv%d.mat',cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot %&& cv==1
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','West');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
            saveas(gcf,['loss curves\TENASPIS loss ',sub1,'_',sub2,' train cv',num2str(cv),'.png']);
        end
    end
end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3)
% if num_nf > num_res
%     [list_nframes',Table_mean]
% elseif num_nf < num_res
%     [list_res',Table_mean]
% end


%% res0_avg drop for simulated data
scale_lowBG = 5e3;
scale_noise = 1;
data_name = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
path_name = fullfile('E:\simulation_CNMFE_corr_noise',data_name);
nvideo = 10;
d0 = 0.8;
list_lam = [5,8,10]; % 3,
doesplot = 0; % _weighted_sum_expanded_edge_unmask
optimize_th_cnn = true; % false; % 
res = 0; % [0,1,3:9]; % 
num_frame = 0; % [0,2:8]; % 
mask_option = 'Xmask'; % {'nomask', 'mask', 'bmask', 'Xmask'}; % 
shuffle = ''; % '_shuffle'; % 
num_mask_channel = 1; % [1,2]; % 
nv = length(list_lam);
Table = zeros(nv,4,nvideo);
for vid=1:length(list_lam)
    lam = list_lam(vid);
    dir_video = fullfile(path_name, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
    dir_parent = fullfile(dir_video,'add_new_blockwise_weighted_sum_unmask'); % GT Masks\
    if strcmp(mask_option,'nomask')
        if num_mask_channel == 1
            vid = vid+1;
        elseif num_mask_channel == 2
            continue;
        end
    end
    if num_frame <= 1
        img_option = 'avg';
        str_shuffle = '';
    else
        img_option = 'multi_frame';
        str_shuffle = shuffle;
    end
    sub1 = sprintf('classifiers_%s_%s',img_option,mask_option);
    sub2 = sprintf('classifier_res%d_%d+%d frames%s',res,num_frame,num_mask_channel,str_shuffle);
    for cv = 1:nvideo
        load(fullfile(dir_parent,sub1,sub2,sprintf('training_output_cv%d.mat',cv-1)));
        if optimize_th_cnn
            Table(vid,:,cv) = [max(list_f1),mean(loss(end-49:end)),mean(accuracy(end-49:end)),th_cnn_best];
        else
            Table(vid,:,cv) = [list_f1((end+1)/2),mean(loss(end-49:end)),mean(accuracy(end-49:end)),0.5];
        end

        if doesplot % && cv==1
            figure; 
            plot(loss); 
            hold on; 
            plot(accuracy); 
            legend({'loss','accuracy'},'Location','Southwest');
            xlabel('Epoch');
            ylim([0,1]);
            set(gca,'FontSize',14);
            saveas(gcf,['loss curves\TENASPIS loss ',sprintf('dropout %gexp(-%g)',d0,lam),'_',sub1,'_',sub2,' train cv',num2str(cv),'.png']);
        end
    end
end
% Table = cell2mat(Table');
Table_mean = mean(Table,3)
% if num_nf > num_res
%     [list_nframes',Table_mean]
% elseif num_nf < num_res
%     [list_res',Table_mean]
% end
