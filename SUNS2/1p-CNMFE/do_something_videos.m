%% play SNR videos and temporal masks of different unmixing methods
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
dir_video = 'E:\1photon-small';
num_Exp = 9;
Exp_ID = list_Exp_ID{9};
%
dir_FISSA='\complete_FISSA\';
SNR_video_FISSA = h5read([dir_video,dir_FISSA,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_FISSA = h5read([dir_video,dir_FISSA,'\temporal_masks(4)\',Exp_ID,'.h5'],'/temporal_masks');
load([dir_video,dir_FISSA,'\test_CNN\4816[1]th4\output_masks pmap\Output_Masks_',Exp_ID,'.mat'],'Masks','pmaps_b','times_active');
Masks_FISSA = permute(Masks,[3,2,1]);
pmaps_b_FISSA = permute(pmaps_b,[3,2,1]);
[Lx, Ly, ~] = size(Masks_FISSA);
cat_video_FISSA = cat(2,SNR_video_FISSA(1:Lx,1:Ly,:),pmaps_b_FISSA,single(temporal_masks_FISSA(1:Lx,1:Ly,:))*7);
%
dir_TUnCaT='\complete_TUnCaT\';
SNR_video_TUnCaT = h5read([dir_video,dir_TUnCaT,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_TUnCaT = h5read([dir_video,dir_TUnCaT,'\temporal_masks(6)\',Exp_ID,'.h5'],'/temporal_masks');
load([dir_video,dir_TUnCaT,'\test_CNN\4816[1]th6\output_masks pmap\Output_Masks_',Exp_ID,'.mat'],'Masks','pmaps_b','times_active');
Masks_TUnCaT = permute(Masks,[3,2,1]);
pmaps_b_TUnCaT = permute(pmaps_b,[3,2,1]);
cat_video_TUnCaT = cat(2,SNR_video_TUnCaT(1:Lx,1:Ly,:),pmaps_b_TUnCaT,single(temporal_masks_TUnCaT(1:Lx,1:Ly,:))*7);
%
cat_video = cat(1,cat_video_FISSA,cat_video_TUnCaT);
figure(99); imshow3D(cat_video,[-0,10]); colorbar;
set(gcf,'Position',[100,100,1100,800]);

