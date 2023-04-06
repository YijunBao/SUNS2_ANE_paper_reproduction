%% play SNR videos and temporal masks of different unmixing methods
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
dir_video = 'E:\1photon-small';
num_Exp = 9;
Exp_ID = list_Exp_ID{5};
%
dir_FISSA='\complete-FISSA\';
SNR_video_FISSA = h5read([dir_video,dir_FISSA,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_FISSA = h5read([dir_video,dir_FISSA,'\temporal_masks(3)\',Exp_ID,'.h5'],'/temporal_masks');
cat_video_FISSA = cat(1,SNR_video_FISSA,single(temporal_masks_FISSA)*5);
%
dir_TUnCaT='\complete-TUnCaT\';
SNR_video_TUnCaT = h5read([dir_video,dir_TUnCaT,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_TUnCaT = h5read([dir_video,dir_TUnCaT,'\temporal_masks(6)\',Exp_ID,'.h5'],'/temporal_masks');
cat_video_TUnCaT = cat(1,SNR_video_TUnCaT,single(temporal_masks_TUnCaT)*5);
%
dir_bgsubs='\complete\';
SNR_video_bgsubs = h5read([dir_video,dir_bgsubs,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_bgsubs = h5read([dir_video,dir_bgsubs,'\temporal_masks(6)\',Exp_ID,'.h5'],'/temporal_masks');
cat_video_bgsubs = cat(1,SNR_video_bgsubs,single(temporal_masks_bgsubs)*5);
%
cat_video = cat(2,cat_video_FISSA,cat_video_TUnCaT,cat_video_bgsubs);
figure(99); imshow3D(cat_video,[-0,5]); colorbar;

%% Count the percentage of temporal masks
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
dir_video = 'E:\1photon-small';
num_Exp = 9;
%
% dir_FISSA='\complete-FISSA\';
% list_thred_ratio = 3:6;
% num_thred_ratio = length(list_thred_ratio);
% list_temporal_masks_mean = zeros(num_Exp,num_thred_ratio);
% for eid=1:9
%     Exp_ID = list_Exp_ID{eid};
%     for thid = 1:num_thred_ratio
%         thred_ratio = list_thred_ratio(thid);
%         temporal_masks = h5read([dir_video,dir_FISSA,'\temporal_masks(',...
%             num2str(thred_ratio),')\',Exp_ID,'.h5'],'/temporal_masks');
%         list_temporal_masks_mean(eid,thid) = mean(temporal_masks,'all')*100;
%     end
% end
% list_temporal_masks_mean_fissa = [list_temporal_masks_mean; nanmean(list_temporal_masks_mean,1); nanstd(list_temporal_masks_mean,1,1)];

%
% dir_FISSA='\complete-TUnCaT\';
% list_thred_ratio = 4:9;
% num_thred_ratio = length(list_thred_ratio);
% list_temporal_masks_mean = zeros(num_Exp,num_thred_ratio);
% for eid=1:9
%     Exp_ID = list_Exp_ID{eid};
%     for thid = 1:num_thred_ratio
%         thred_ratio = list_thred_ratio(thid);
%         temporal_masks = h5read([dir_video,dir_FISSA,'\temporal_masks(',...
%             num2str(thred_ratio),')\',Exp_ID,'.h5'],'/temporal_masks');
%         list_temporal_masks_mean(eid,thid) = mean(temporal_masks,'all')*100;
%     end
% end
% list_temporal_masks_mean_tuncat = [list_temporal_masks_mean; nanmean(list_temporal_masks_mean,1); nanstd(list_temporal_masks_mean,1,1)];

%
dir_FISSA='\complete\';
list_thred_ratio = 3:6;
num_thred_ratio = length(list_thred_ratio);
list_temporal_masks_mean = zeros(num_Exp,num_thred_ratio);
for eid=1:9
    Exp_ID = list_Exp_ID{eid};
    for thid = 1:num_thred_ratio
        thred_ratio = list_thred_ratio(thid);
        temporal_masks = h5read([dir_video,dir_FISSA,'\temporal_masks(',...
            num2str(thred_ratio),')\',Exp_ID,'.h5'],'/temporal_masks');
        list_temporal_masks_mean(eid,thid) = mean(temporal_masks,'all')*100;
    end
end
list_temporal_masks_mean_bgsubs = [list_temporal_masks_mean; nanmean(list_temporal_masks_mean,1); nanstd(list_temporal_masks_mean,1,1)];

% save('temporal_masks_mean.mat','list_temporal_masks_mean_fissa','list_temporal_masks_mean_tuncat','list_temporal_masks_mean_bgsubs')
