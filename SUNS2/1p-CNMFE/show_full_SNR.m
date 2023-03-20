addpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools');
dir_video='E:\OnePhoton videos\full videos\';
dir_SNR = [dir_video,'complete-FISSA\network_input\'];
dir_masks = [dir_video,'GT Masks\'];
list_Exp_ID = {'c25_1NRE','c27_NN','c28_1NRE1NLE'};
list_names = {'c25_59_228','c27_12_326','c28_83_210';
            'c25_163_267','c27_114_176','c28_161_149';
            'c25_123_348','c27_122_121','c28_163_244'};
[num_each,num_Exp] = size(list_names);
% list_h5name = {'c25_1NRE_20Hz_35min_registerednon-rigid_sbinned_normalized.h5',...
%         'c27_NN_20Hz_35min_registerednon-rigid_sbinned_normalized.h5',...
%         'c28_1NRE1NLE_20Hz_35min_registerednon-rigid_sbinned_normalized.h5'};
% list_maskname = {'c25_1NRE_20Hz_35min_bandpassed_foundMasks.mat',...
%                 'c27_NN_20Hz_35min_bandpassed_foundMasks_Alissa.mat'...
%                 'c28_1NRE1NLE_20Hz_35min_bandpassed_foundMasks_Alissa.mat'};

eid = 1;

%%
regvideo = h5read([dir_SNR,list_Exp_ID{eid},'.h5'],'/network_input');
video_max = max(regvideo,[],3);
video_min = min(regvideo,[],3);
video_mean = mean(regvideo,3);
video_std = std(single(regvideo),1,3);

%%
load([dir_masks,'FinalMasks_',list_Exp_ID{eid},'.mat'],'FinalMasks')
% FinalMasks = wholeMask > 0.2 * max(max(wholeMask,1),2);
sumMask = sum(FinalMasks,3);
% plot_masks_id(FinalMasks,wholeMask);

%%
figure('Position',[0,0,1920,1080]); 
imagesc(video_std); colorbar; axis('image');
hold on;
contour(sumMask);
title('Standard deviation');
% saveas(gcf,['std ',list_Exp_ID{eid},'.png']);
%%
figure('Position',[0,0,1920,1080]); 
imagesc(video_max); colorbar; axis('image');
hold on;
ncells = size(FinalMasks,3);
for n = 1:ncells
    mask = FinalMasks(:,:,n);
    contour(mask,'r');
end
% contour(sumMask);
title('Max projection');
% saveas(gcf,['max ',list_Exp_ID{eid},'.png']);

%%
for cid = 1:num_each
    crop_name = list_names{cid,eid};
    crop_name_list = split(crop_name,'_');
    xstart = str2double(crop_name_list{2});
    ystart = str2double(crop_name_list{3});
    rectangle('Position',[ystart,xstart,50,50],'LineWidth',2,'EdgeColor','k');
end
% saveas(gcf,['crop ',list_Exp_ID{eid},'.png']);
