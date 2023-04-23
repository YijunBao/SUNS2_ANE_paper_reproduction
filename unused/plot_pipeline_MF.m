color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% green = [0.1,0.9,0.1]; % color(5,:); %
% red = [0.9,0.1,0.1]; % color(7,:); %
% blue = [0.1,0.8,0.9]; % color(6,:); %
yellow = [0.8,0.8,0.0]; % color(3,:); %
magenta = [0.9,0.3,0.9]; % color(4,:); %
green = [0.0,0.65,0.0]; % color(5,:); %
red = [0.8,0.0,0.0]; % color(7,:); %
blue = [0.0,0.6,0.8]; % color(6,:); %
colors_multi = distinguishable_colors(16);

alpha = 0.8;
addpath(genpath('C:\Matlab Files\missing_finder'))

%%
% list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
%              'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
% num_Exp = length(list_Exp_ID);
% dir_video = 'D:\data_TENASPIS\added_refined_masks';
% eid = 1;
% Exp_ID = list_Exp_ID{eid};

% load(['TENASPIS mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
% load(['TENASPIS mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');

%% 
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_title = {'upper left', 'upper right', 'lower left', 'lower right'};
list_avg_radius = [5,6,8,14];
data_ind = 4;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\data_CNMFE',data_name);
dir_video_raw = dir_video;
dir_video_SNR = fullfile(dir_video,'complete_TUnCaT_SF50\network_input\');
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
eid=3;
Exp_ID = list_Exp_ID{eid};

save_folder = '.\plot pipeline\';
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

%% load results
video_raw = h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),'/mov'); % raw_traces
raw_max = max(video_raw,[],3);
video_SNR = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input'); % raw_traces
SNR_max = max(video_SNR,[],3);

dir_SUNS = fullfile(dir_video, 'complete_TUnCaT_SF50\4816[1]th4'); % 4 v1
dir_masks = fullfile(dir_SUNS, 'output_masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');
load(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
    'masks'); % ,'traces_raw','sum_edges','video'
%     'list_weight','list_weight_trace','list_weight_frame',...
load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
    'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
    'images_added_crop', 'patch_locations','time_weights','masks_neighbors_crop'); % ,'list_valid'

dir_CNN_pred = fullfile(dir_add_new,'trained dropout 0.8exp(-5)\avg_Xmask_0.5\classifier_res0_0+1 frames');
load(fullfile(dir_CNN_pred,['CNN_predict_',Exp_ID,'_cv',num2str(eid-1),'.mat']),'pred_valid','time_CNN');

dir_GT_masks = fullfile(dir_video,'GT Masks');
load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
temp = [patch_locations, pred_valid'];

%%
mag=2;
mag_kernel = ones(mag,mag,class(images_added_crop));
mag_kernel_uint8 = ones(mag,mag,'uint8');
mag_kernel_bool = logical(mag_kernel);

%%
avg_radius = list_avg_radius(data_ind);
r_bg_ratio = 3;
leng = r_bg_ratio*avg_radius;

select_k = 13;
locations_select = patch_locations(select_k,:);
list_k = find(~any(patch_locations - locations_select,2));
pred_select = pred_valid(list_k);
xmin = locations_select(1);
xmax = locations_select(2);
ymin = locations_select(3);
ymax = locations_select(4);
ix = ceil((xmin-2)/leng)+1;
iy = ceil((ymin-2)/leng)+1;
% weight_select = list_weight{ix,iy};
SNR_max_crop = SNR_max(xmin:xmax,ymin:ymax);
SNR_max_crop_mag = kron(SNR_max_crop,mag_kernel);

% [Lx,Ly,N] = size(masks);
masks_sub = masks(xmin:xmax,ymin:ymax,:);
neighbors = squeeze(sum(sum(masks_sub,1),2)) > 0;
masks_neighbors = masks_sub(:,:,neighbors);
% [Lxm,Lym,num_neighbors] = size(masks_neighbors);

masks_GT_sub = FinalMasks(xmin:xmax,ymin:ymax,:);
neighbors_GT = squeeze(sum(sum(masks_GT_sub,1),2)) > 0;
masks_neighbors_GT = masks_GT_sub(:,:,neighbors_GT);

added_frames_select = added_frames(list_k);
added_weights_select = added_weights(list_k);
images_added_crop_select = images_added_crop(:,:,list_k);
masks_added_crop_select = masks_added_crop(:,:,list_k);
% masks_neighbors_crop_select = masks_neighbors_crop(:,:,list_k);

%%
[Lxm,Lym,num_neighbors] = size(masks_neighbors);
% edge_masks_neighbors = 0*masks_neighbors;
% for nn = 1:size(masks_neighbors,3)
%     edge_masks_neighbors(:,:,nn) = edge(masks_neighbors(:,:,nn));
% end
% masks_neighbors_sum = sum(edge_masks_neighbors,3);
masks_neighbors_sum = sum(masks_neighbors,3);
masks_neighbors_sum_mag = kron(masks_neighbors_sum,mag_kernel);

%%
% [Lxm,Lym,num_neighbors] = size(masks_neighbors_GT);
% edge_masks_neighbors_GT = 0*masks_neighbors_GT;
% for nn = 1:size(masks_neighbors_GT,3)
%     edge_masks_neighbors_GT(:,:,nn) = edge(masks_neighbors_GT(:,:,nn));
% end
% masks_neighbors_GT_sum = sum(edge_masks_neighbors_GT,3);
masks_neighbors_GT_sum = sum(masks_neighbors_GT,3);
masks_neighbors_GT_sum_mag = kron(masks_neighbors_GT_sum,mag_kernel);

%% plot SUNS masks on peak SNR image
figure; imshow(SNR_max_crop_mag,[2,14]); % colormap gray;
hold on;
% alphaSUNS = ones(Lxm*mag,Lym*mag).*reshape(color(5,:),1,1,3);
% image(alphaSUNS,'Alphadata',alpha*(masks_neighbors_sum_mag));  
contour(masks_neighbors_sum_mag,'EdgeColor',color(5,:),'LineWidth',1); % ,magenta
rectangle('Position',[5,10,20*mag,4*mag],'FaceColor','w','LineStyle','None'); % 20 um scale bar
frame=getframe(gcf);
cdata_initial=frame.cdata;
% cdata_initial_mag=zeros(Lxm*mag,Lym*mag,3,'uint8');
% for kc=1:3
%     cdata_initial_mag(:,:,kc)=kron(cdata_initial(:,:,kc),mag_kernel_uint8);
% end
imwrite(cdata_initial,[save_folder, 'Masks_SUNS crop ',Exp_ID,' ',mat2str(locations_select),'.png']);

masks_added_crop_mag = kron((sum(masks_added_crop(:,:,list_k(pred_select)),3)),mag_kernel_bool);
% masks_added_crop_mag = kron(edge(sum(masks_added_crop(:,:,list_k(pred_select)),3)),mag_kernel_bool);
% alphaImg = ones(Lxm*mag,Lym*mag).*reshape(color(6,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(masks_added_crop_mag));  
contour(masks_added_crop_mag,'EdgeColor',color(6,:),'LineWidth',1); % ,magenta
frame=getframe(gcf);
cdata_initial=frame.cdata;
imwrite(cdata_initial,[save_folder, 'Masks_SUNS+MF crop ',Exp_ID,' ',mat2str(locations_select),'.png']);

% alphaImg = ones(Lxm*mag,Lym*mag).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(masks_neighbors_GT_sum_mag));  
contour(masks_neighbors_GT_sum_mag,'EdgeColor',color(3,:),'LineWidth',1); % ,magenta
frame=getframe(gcf);
cdata_initial=frame.cdata;
imwrite(cdata_initial,[save_folder, 'Masks_SUNS+MF+GT crop ',Exp_ID,' ',mat2str(locations_select),'.png']);

%% plot missing masks on averaged weighted images
% list_crange = [0,3; 0,4; 0,5];
for ik = 1:length(list_k)
    k = list_k(ik);
    images_added_crop_mag = kron(images_added_crop(:,:,k),mag_kernel);
    masks_added_crop_mag = kron((masks_added_crop(:,:,k)),mag_kernel_bool);
%     masks_added_crop_mag = kron(edge(masks_added_crop(:,:,k)),mag_kernel_bool);

%     figure; imshow(images_added_crop_mag,list_crange(ik,:)); % colormap gray;
    figure; imshow(images_added_crop_mag,[-2,4]); % colormap gray;
    hold on;
%     image(alphaSUNS,'Alphadata',alpha*(masks_neighbors_sum_mag));  
%     alphaImg = ones(Lxm*mag,Lym*mag).*reshape(color(6,:),1,1,3);
%     image(alphaImg,'Alphadata',alpha*(masks_added_crop_mag));  
    contour(masks_neighbors_sum_mag,'EdgeColor',color(5,:),'LineWidth',1);
    contour(masks_added_crop_mag,'EdgeColor',color(6,:),'LineWidth',1);
    text(6,Lxm*mag-24,num2str(ik),'FontSize',28,'Color','y');
    if ik == 1
        rectangle('Position',[5,10,20*mag,4*mag],'FaceColor','w','LineStyle','None'); % 20 um scale bar
    end
    frame=getframe(gcf);
    cdata_added=frame.cdata;
    % cdata_initial_mag=zeros(Lxm*mag,Lym*mag,3,'uint8');
    % for kc=1:3
    %     cdata_initial_mag(:,:,kc)=kron(cdata_initial(:,:,kc),mag_kernel_uint8);
    % end
    imwrite(cdata_added,[save_folder, 'Masks_MF ',Exp_ID,' ',mat2str(k),'-',num2str(pred_select(ik)),'.png']);
end

%% plot the tiled selected frames
load(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
    'list_weight','video','masks');
weight = list_weight{ix,iy};
T = size(video,3);
num_avg = max(60, min(90, ceil(T*0.01)));
th_IoU_split = 0.5;
thj_inclass = 0.4;
thj = 0.7;
area = squeeze(sum(sum(masks,1),2));
avg_area = median(area);
%%
[image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss, ...
    classes, video_sub_sort, mask_update_select, select_frames_sort_full] = ...
    find_missing_blockwise_class(video, masks, xmin, xmax, ymin, ymax, ...
    weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split);

%%
tile_size = [4,6];
images_tile = imtile(video_sub_sort, 'GridSize', tile_size);
masks_tile = imtile(mask_update_select, 'GridSize', tile_size);

figure;
imshow(images_tile,[-2,4]);
hold on;
contour(masks_tile,'EdgeColor',red,'LineWidth',1);
for kk = 1:length(classes)
    [yn,xn] = ind2sub(fliplr(tile_size),kk);
    xx = (xn)*Lxm-16;
%     xx = (xn-1)*Lxm+12;
    yy = (yn-1)*Lym+6;
    if classes(kk)
        text(yy,xx,num2str(classes(kk)),'FontSize',18,'Color','y');
%         text(yy,xx,[num2str(classes(kk)),':  ',num2str(select_frames_sort_full(kk))],'FontSize',18,'Color','y');
    else
        text(yy,xx,'X','FontSize',18,'Color','y');
%         text(yy,xx,['X:  ',num2str(select_frames_sort_full(kk))],'FontSize',18,'Color','y');
    end
end
% %% 
for xn = 0:tile_size(1)
    plot([0,tile_size(2)*Lym],[xn*Lxm,xn*Lxm],'w','LineWidth',2);
end
for yn = 0:tile_size(2)
    plot([yn*Lym,yn*Lym],[0,tile_size(1)*Lxm],'w','LineWidth',2);
end
% %%
rectangle('Position',[size(masks_tile,2)-30,size(masks_tile,1)-10,20,4],'FaceColor','w','LineStyle','None'); % 20 um scale bar
% %%
% alphaImg = ones(size(images_tile)).*reshape(red,1,1,3);
% image(alphaImg,'Alphadata',alpha*(masks_tile));  
frame=getframe(gcf);
cdata_initial=frame.cdata;
imwrite(cdata_initial,[save_folder, 'Frame_tiles ',Exp_ID,' ',mat2str(locations_select),'.png']);

