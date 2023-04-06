%% play SNR videos and temporal masks of different unmixing methods
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
dir_video = 'E:\1photon-small';
num_Exp = 9;
Exp_ID = list_Exp_ID{num_Exp};
%
dir_FISSA='\complete_FISSA\';
SNR_video_FISSA = h5read([dir_video,dir_FISSA,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_FISSA = h5read([dir_video,dir_FISSA,'\temporal_masks(4)\',Exp_ID,'.h5'],'/temporal_masks');
load([dir_video,dir_FISSA,'\test_CNN\4816[1]th4\output_masks pmap\Output_Masks_',Exp_ID,'.mat'],'Masks','pmaps_b','times_active');
Masks_FISSA = permute(Masks,[3,2,1]);
pmaps_b_FISSA = permute(pmaps_b,[3,2,1]);
[Lx, Ly, ~] = size(Masks_FISSA);
% cat_video_FISSA = cat(2,SNR_video_FISSA(1:Lx,1:Ly,:),pmaps_b_FISSA,single(temporal_masks_FISSA(1:Lx,1:Ly,:))*7);
%
dir_TUnCaT='\complete_TUnCaT\';
SNR_video_TUnCaT = h5read([dir_video,dir_TUnCaT,'\network_input\',Exp_ID,'.h5'],'/network_input');
temporal_masks_TUnCaT = h5read([dir_video,dir_TUnCaT,'\temporal_masks(6)\',Exp_ID,'.h5'],'/temporal_masks');
load([dir_video,dir_TUnCaT,'\test_CNN\4816[1]th6\output_masks pmap\Output_Masks_',Exp_ID,'.mat'],'Masks','pmaps_b','times_active');
Masks_TUnCaT = permute(Masks,[3,2,1]);
pmaps_b_TUnCaT = permute(pmaps_b,[3,2,1]);
% cat_video_TUnCaT = cat(2,SNR_video_TUnCaT(1:Lx,1:Ly,:),pmaps_b_TUnCaT,single(temporal_masks_TUnCaT(1:Lx,1:Ly,:))*7);
%%
% cat_video = cat(1,cat_video_FISSA,cat_video_TUnCaT);
% figure(99); imshow3D(cat_video,[-0,10]); colorbar;
t = 3959;
SNR_max = 10;
figure('Position',[100,100,1100,700]);

subplot(2,3,1);
imagesc(SNR_video_FISSA(1:Lx,1:Ly,t),[0,SNR_max]);
axis('image','off'); % colorbar;
title(['SNR video (0-',num2str(SNR_max),')'],'FontSize',12);

subplot(2,3,2);
imagesc(pmaps_b_FISSA(1:Lx,1:Ly,t),[0,10]);
axis('image','off'); % colorbar;
title('Probablity map FISSA','FontSize',12);

subplot(2,3,3);
imagesc(temporal_masks_FISSA(1:Lx,1:Ly,t),[0,1.5]);
axis('image','off'); % colorbar;
title('Temporal masks FISSA','FontSize',12);

subplot(2,3,4);
imagesc(SNR_video_TUnCaT(1:Lx,1:Ly,t),[0,SNR_max]);
axis('image','off'); % colorbar;
title(['SNR video (0-',num2str(SNR_max),')'],'FontSize',12);

subplot(2,3,5);
imagesc(pmaps_b_TUnCaT(1:Lx,1:Ly,t),[0,10]);
axis('image','off'); % colorbar;
title('Probablity map TUnCaT','FontSize',12);

subplot(2,3,6);
imagesc(temporal_masks_TUnCaT(1:Lx,1:Ly,t),[0,1.5]);
axis('image','off'); % colorbar;
title('Temporal masks TUnCaT','FontSize',12);

suptitle([replace(Exp_ID,'_','-'),', Frame ',num2str(t)])
saveas(gcf,[Exp_ID,', Frame ',num2str(t),'.png'])


%%
comx=zeros(1,N);
comy=zeros(1,N);
for nn=1:N
    [xxs,yys]=find(FinalMasks(:,:,nn)>0);
    comx(nn)=mean(xxs);
    comy(nn)=mean(yys);
end
max_SNR = max(video_SNR,[],3);
%%
figure;
n = 2;
% t = 1595;
% [val,t] = max(d(n,:));
t = 2;
imagesc(video_SNR(:,:,t));
% imagesc(max_SNR);
axis('image');
colormap gray;
hold on; 
mask = FinalMasks(:,:,n);
r_bg=sqrt((sum(sum(mask)))/pi)*2.5;
r = sqrt((comx(n)-comx).^2 + (comy(n)-comy).^2); 
neighbors = setdiff(find(r < r_bg),n); 
for nn = 1:length(neighbors)
    contour(FinalMasks(:,:,neighbors(nn)),'y');
end
neighbors = list_neighbors{n}; 
for nn = 1:length(neighbors)
    contour(FinalMasks(:,:,neighbors(nn)),'g');
end
contour(mask,'r');

%%
figure;
for t = 1:find(select_frames)
    imagesc(video_sub(:,:,t));
    colormap gray;
    colorbar;
    axis('image');
    hold on;
    contour(mask_update,'r');
end
%%
figure;
imagesc(avg_frame);
colormap gray;
colorbar;
axis('image');
hold on;
contour(mask_sub,'g');
contour(mask_update,'r');

