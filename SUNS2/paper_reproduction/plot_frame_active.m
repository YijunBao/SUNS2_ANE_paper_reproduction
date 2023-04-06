mag=2;
mag_kernel = ones(mag,mag);
%% play SNR videos and temporal masks of different unmixing methods
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
dir_video = 'E:\1photon-small';
num_Exp = 8;
Exp_ID = list_Exp_ID{num_Exp};
%
dir_TUnCaT='\complete_TUnCaT\';
SNR_video_TUnCaT = h5read([dir_video,dir_TUnCaT,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_TUnCaT = h5read([dir_video,dir_TUnCaT,'\temporal_masks(6)\',Exp_ID,'.h5'],'/temporal_masks');
load([dir_video,dir_TUnCaT,'\test_CNN\4816[1]th6\output_masks pmap\Output_Masks_',Exp_ID,'.mat'],'Masks','pmaps_b','times_active');
Masks_TUnCaT = permute(Masks,[3,2,1]);
pmaps_b_TUnCaT = permute(pmaps_b,[3,2,1]);
[Lx,Ly,ncells] = size(Masks_TUnCaT);
%%
t = 3421;
figure('Position',[100,100,400,300]);
imagesc(SNR_video_TUnCaT(1:Lx,1:Ly,t),[0,5]);
axis('image','off'); 
h=colorbar;
set(h,'FontSize',12);
set(get(h,'Label'),'String','SNR','FontName','Arial');
colormap gray;
title(['Frame ',num2str(t)],'FontSize',12);
hold on;
cells_active = cellfun(@(x) any(x==t), times_active);
temp_mask = sum(Masks_TUnCaT(:,:,cells_active),3);
temp_mask_mag = kron(temp_mask,mag_kernel);
contour(temp_mask,'Color', [0.9,0.1,0.1]);
% saveas(gcf,[Exp_ID,', Frame ',num2str(t),' active.png'])

%% show masks with ID
% addpath('C:\Users\Yijun\OneDrive\NeuroToolbox\Matlab files\plot tools');
% for eid=1:9
%     Exp_ID = list_Exp_ID{eid};
%     load([dir_video,dir_TUnCaT,'\test_CNN\4816[1]th6\output_masks pmap\Output_Masks_',Exp_ID,'.mat'],'Masks');
%     Masks_TUnCaT = permute(Masks,[3,2,1]);
%     plot_masks_id(Masks_TUnCaT);
%     axis('image','off');
%     title(Exp_ID,'interpreter','none');
%     saveas(gcf,[Exp_ID,'_SUNS_TUnCaT output.png'])
% end