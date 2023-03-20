clear;
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
rate_hz = [10,15,7.5,5]; % frame rate of each video
radius = [5,6,6,6];

dir_video = 'E:\data_CNMFE\full videos';
% num_Exp = length(list_Exp_ID);

% dir_sub='complete_TUnCaT\4816[1]th4'; %_2out+BGlayer
dir_SNR='complete_TUnCaT\network_input'; %_2out+BGlayer
dir_temporal_masks='complete_TUnCaT\temporal_masks(4)'; %_2out+BGlayer
dir_GT_masks=fullfile(dir_video,'GT Masks');
% dir_output_masks=fullfile(dir_video,dir_sub,'output_masks pmap');

for eid=1:2%;
Exp_ID = list_Exp_ID{eid};

filter_tempolate = h5read(fullfile(dir_video,[Exp_ID,'_spike_tempolate.h5']),'/filter_tempolate');
[~,peak] = max(filter_tempolate);
[~,rise] = find(filter_tempolate > exp(-1),1);
lag = peak - rise;

nframes = 100;
fps = rate_hz(eid);
start=[1,1,400];
count=[Inf,Inf,nframes];
stride=[1,1,1];

% color_range_raw = [0,200];
color_range_raw = [1000,2500];
% Lx=list_patch_dims(eid,1); 
% Ly=list_patch_dims(eid,2);

video_raw = h5read(fullfile(dir_video,[Exp_ID,'.h5']),'/mov',start+[0,0,lag], count, stride);
% video_raw_min = min(video_raw,[],3);
% video_raw = video_raw - video_raw_min;
[Lx,Ly,T] = size(video_raw);
color_range_raw = [min(video_raw,[],'all'),max(video_raw,[],'all')];

load(fullfile(dir_GT_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
% load(fullfile(dir_output_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks','times_active');
% Masks = permute(Masks,[3,2,1]);

rangex=1:Ly; 
rangey=1:Lx;
crop_img=[86,64,Ly,Lx];

%% Output on raw video
% % border = 255*ones(Lxc,10,3,'uint8');
% v = VideoWriter([Exp_ID,'Masks 2 raw.avi']);
% v.FrameRate = fps;
% open(v);
% figure('Position',[100,100,680,360],'Color','w');
% 
% for t = 1:nframes
%     t_real = start(3) + t - 1;
%     image = video_raw(:,:,t);
%     
%     clf; % masks from SUNS online
%     imshow(image(rangey,rangex)', color_range_raw); % 
%     neurons_active = cellfun(@(x) ~isempty(find(x==t_real,1)), times_active);
%     mask = sum(Masks(:,:,neurons_active),3);
% 
%     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
%     h=colorbar;
%     set(h,'FontSize',9);
%     set(get(h,'Label'),'String','Raw intensity','FontName','Arial');
%     title('Output on raw video')
%     hold on;
%     contour(mask(rangey,rangex)','Color', [0.9,0.1,0.1]);
%     pause(0.001);
%     img_all=getframe(gcf); % ,crop_img
%     img_notrack=img_all.cdata;
%     
% %     figure(99); imshow(img_both);
%     writeVideo(v,img_notrack);
% end
% close(v);


%% Output on SNR video
% color_range_SNR = [-2,5]; % [0,3];
% video_SNR = h5read(fullfile(dir_video,dir_SNR,[Exp_ID,'.h5']),'/network_input',start, count, stride);
% temporal_masks = h5read(fullfile(dir_video,dir_temporal_masks,[Exp_ID,'.h5']),'/temporal_masks',start, count, stride);
% load(fullfile(dir_output_masks,[Exp_ID,'_pmap.mat']),'prob_map');
% prob_map = permute(prob_map,[3,2,1]);
% prob_map = prob_map(:,:,start(3):start(3)+count(3)-1);
% % %%
% v = VideoWriter([Exp_ID,'GT SNR and SUNS pmap [-2,5].avi']);
% v.FrameRate = fps;
% open(v);
% figure('Position',[100,100,780,400],'Color','w');
% 
% for t = 1:nframes
%     t_real = start(3) + t - 1;
%     image = video_SNR(:,:,t);
%     image = image(rangey,rangex)';
% %     image_show = [image,image];
% %     clf; % masks from SUNS online
% %     imshow(image, color_range_SNR);
%     pmap = prob_map(:,:,t);
%     pmap = pmap(rangey,rangex)';
%     
%     mask1 = temporal_masks(:,:,t);
%     mask1 = mask1(rangey,rangex)';
%     neurons_active = cellfun(@(x) ~isempty(find(x==t_real,1)), times_active);
%     mask2 = sum(Masks(:,:,neurons_active),3);
%     mask2 = mask2(rangey,rangex)';
% %     mask1_show = [mask1, 0*mask2];
% %     mask2_show = [0*mask1, mask2];
% 
%     subplot(1,2,1);
%     cla;
%     imshow(image, color_range_SNR);
%     hold on;
%     contour(mask1,'Color', [0.1,0.9,0.1]);
%     title('GT on SNR video')
% 
%     subplot(1,2,2);
%     cla;
%     imshow(pmap, [0,1]);
%     hold on;
%     contour(mask2,'Color', [0.9,0.1,0.1]);
%     title('SUNS on probability map')
% 
% %     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
% %     h=colorbar;
% %     set(h,'FontSize',9);
% %     set(get(h,'Label'),'String','SNR','FontName','Arial');
% %     title('GT (left) and Output (right) on SNR video')
% %     hold on;
% %     contour(mask1_show,'Color', [0.1,0.9,0.1]);
% %     contour(mask2_show,'Color', [0.9,0.1,0.1]);
%     pause(0.001);
%     img_all=getframe(gcf); % ,crop_img
%     img_notrack=img_all.cdata;
% 
% %     figure(99); imshow(img_both);
%     writeVideo(v,img_notrack);
% end
% close(v);


%% GT on Raw video
temporal_masks = h5read(fullfile(dir_video,dir_temporal_masks,[Exp_ID,'.h5']),'/temporal_masks',start, count, stride);

v = VideoWriter([Exp_ID,' no Masks raw.avi']);
v.FrameRate = fps;
open(v);
% figure('Position',[100,100,680,560],'Color','w');

raw_min = single(min(video_raw,[],'all'));
raw_max = single(max(video_raw,[],'all'));
raw_range = raw_max - raw_min;

for t = 1:nframes
    t_real = start(3) + t - 1;
    img = single(video_raw(:,:,t));
    img = (img - raw_min)/raw_range;
    writeVideo(v,single(img));
end
close(v);


%% GT on Raw video
% temporal_masks = h5read(fullfile(dir_video,dir_temporal_masks,[Exp_ID,'.h5']),'/temporal_masks',start, count, stride);
% 
% v = VideoWriter([Exp_ID,' GT Masks raw.avi']);
% v.FrameRate = fps;
% open(v);
% figure('Position',[100,100,680,560],'Color','w');
% 
% for t = 1:nframes
%     t_real = start(3) + t - 1;
%     img = video_raw(:,:,t);
% %     img = imresize(img,2);
%     
%     clf; % masks from SUNS online
%     imshow(img(rangey,rangex)', color_range_raw);
%     mask = temporal_masks(:,:,t);
% 
% %     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
% %     h=colorbar;
% %     set(h,'FontSize',9);
% %     set(get(h,'Label'),'String','SNR','FontName','Arial');
% %     title('GT temporal masks')
%     hold on;
% %     contour(mask(rangey,rangex)','Color', [0.1,0.9,0.1]);
%     pause(0.001);
%     img_all=getframe(gcf); % ,crop_img
%     img_notrack=img_all.cdata;
% 
% %     figure(99); imshow(img_both);
%     writeVideo(v,img_notrack);
% end
% close(v);


%% GT on SNR video
% temporal_masks = h5read(fullfile(dir_video,dir_temporal_masks,[Exp_ID,'.h5']),'/temporal_masks',start, count, stride);
% 
% v = VideoWriter([Exp_ID,'GT Masks SNR.avi']);
% v.FrameRate = fps;
% open(v);
% figure('Position',[100,100,680,360],'Color','w');
% 
% for t = 1:nframes
%     t_real = start(3) + t - 1;
%     image = video_SNR(:,:,t);
%     
%     clf; % masks from SUNS online
%     imshow(image(rangey,rangex)', color_range_SNR);
%     mask1 = temporal_masks(:,:,t);
% 
%     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
%     h=colorbar;
%     set(h,'FontSize',9);
%     set(get(h,'Label'),'String','SNR','FontName','Arial');
%     title('GT temporal masks')
%     hold on;
%     contour(mask(rangey,rangex)','Color', [0.1,0.9,0.1]);
%     pause(0.001);
%     img_all=getframe(gcf); % ,crop_img
%     img_notrack=img_all.cdata;
% 
% %     figure(99); imshow(img_both);
%     writeVideo(v,img_notrack);
% end
% close(v);

end