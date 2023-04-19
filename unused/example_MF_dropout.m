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


%% Plot final masks with color coding
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
dir_video = 'D:\data_TENASPIS\added_refined_masks';
eid = 1;
Exp_ID = list_Exp_ID{eid};
alpha = 0.8;

load(['C:\Matlab Files\SUNS-1p\1p-CNMFE\TENASPIS mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
load(['C:\Matlab Files\SUNS-1p\1p-CNMFE\TENASPIS mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');

%% load results
dir_masks = fullfile(dir_video, 'GT Masks dropout 0.8exp(-15)'); % 4 v1
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');
load(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),'masks'); % ,'traces_raw','sum_edges','video'
%     'list_weight','list_weight_trace','list_weight_frame',...
load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']),'patch_locations'); % 
%     'added_frames','added_weights', 'masks_added_full','masks_added_crop','images_added_crop', ...
load(fullfile(dir_add_new,[Exp_ID,'_added_CNNtrain_blockwise.mat']), ...
    'added_frames','added_weights', 'list_valid','masks_added_crop',...
    'images_added_crop', 'masks_neighbors_crop'); % 
% 
% dir_CNN_pred = fullfile(dir_add_new,'trained dropout 0.8exp(-15)\avg_Xmask_0.5\classifier_res0_0+1 frames');
% load(fullfile(dir_CNN_pred,['CNN_predict_',Exp_ID,'_cv',num2str(eid-1),'.mat']),'pred_valid','time_CNN');

dir_GT_masks = fullfile(dir_video,'GT Masks');
load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
temp = [patch_locations, list_valid'];

%%
mag=4;
mag_kernel = ones(mag,mag,class(images_added_crop));
mag_kernel_uint8 = ones(mag,mag,'uint8');
mag_kernel_bool = logical(mag_kernel);

%%
avg_radius = 9;
r_bg_ratio = 3;
leng = r_bg_ratio*avg_radius;

select_k = 27;
locations_select = patch_locations(select_k,:);
list_k = find(~any(patch_locations - locations_select,2));
pred_select = list_valid(list_k);
xmin = locations_select(1);
xmax = locations_select(2);
ymin = locations_select(3);
ymax = locations_select(4);
ix = floor(xmin/leng);
iy = floor(ymin/leng);
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
edge_masks_neighbors = 0*masks_neighbors;
for nn = 1:size(masks_neighbors,3)
    edge_masks_neighbors(:,:,nn) = edge(masks_neighbors(:,:,nn));
end
masks_neighbors_sum = sum(edge_masks_neighbors,3);
masks_neighbors_sum_mag = kron(masks_neighbors_sum,mag_kernel);

%%
% [Lxm,Lym,num_neighbors] = size(masks_neighbors_GT);
edge_masks_neighbors_GT = 0*masks_neighbors_GT;
for nn = 1:size(masks_neighbors_GT,3)
    edge_masks_neighbors_GT(:,:,nn) = edge(masks_neighbors_GT(:,:,nn));
end
masks_neighbors_GT_sum = sum(edge_masks_neighbors_GT,3);
masks_neighbors_GT_sum_mag = kron(masks_neighbors_GT_sum,mag_kernel);

%% plot SUNS masks on peak SNR image
figure; imshow(SNR_max_crop_mag,[2,14]); % colormap gray;
hold on;
alphaSUNS = ones(Lxm*mag,Lym*mag).*reshape(color(5,:),1,1,3);
image(alphaSUNS,'Alphadata',alpha*(masks_neighbors_sum_mag));  
% contour(Masks_initial_sum,'r'); % ,magenta
frame=getframe(gcf);
cdata_initial=frame.cdata;
% cdata_initial_mag=zeros(Lxm*mag,Lym*mag,3,'uint8');
% for kc=1:3
%     cdata_initial_mag(:,:,kc)=kron(cdata_initial(:,:,kc),mag_kernel_uint8);
% end
imwrite(cdata_initial,['Masks_GT crop ',Exp_ID,' ',mat2str(locations_select),'.png']);

alphaImg = ones(Lxm*mag,Lym*mag).*reshape(color(6,:),1,1,3);
masks_added_crop_mag = kron(edge(sum(masks_added_crop(:,:,list_k(pred_select)),3)),mag_kernel_bool);
image(alphaImg,'Alphadata',alpha*(masks_added_crop_mag));  
frame=getframe(gcf);
cdata_initial=frame.cdata;
imwrite(cdata_initial,['Masks_GT+MF crop ',Exp_ID,' ',mat2str(locations_select),'.png']);

alphaImg = ones(Lxm*mag,Lym*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(masks_neighbors_GT_sum_mag));  
frame=getframe(gcf);
cdata_initial=frame.cdata;
imwrite(cdata_initial,['Masks_GT+MF+GT crop ',Exp_ID,' ',mat2str(locations_select),'.png']);

%% plot missing masks on averaged weighted images
list_crange = [0,3; 0,4; 0,5];
for ik = 1:length(list_k)
    k = list_k(ik);
    images_added_crop_mag = kron(images_added_crop(:,:,k),mag_kernel);
    masks_added_crop_mag = kron(edge(masks_added_crop(:,:,k)),mag_kernel_bool);

%     figure; imshow(images_added_crop_mag,list_crange(ik,:)); % colormap gray;
    figure; imshow(images_added_crop_mag,[]); % colormap gray;
    hold on;
    image(alphaSUNS,'Alphadata',alpha*(masks_neighbors_sum_mag));  
    alphaImg = ones(Lxm*mag,Lym*mag).*reshape(color(6,:),1,1,3);
    image(alphaImg,'Alphadata',alpha*(masks_added_crop_mag));  
    % contour(Masks_initial_sum,'r'); % ,magenta
    frame=getframe(gcf);
    cdata_added=frame.cdata;
    % cdata_initial_mag=zeros(Lxm*mag,Lym*mag,3,'uint8');
    % for kc=1:3
    %     cdata_initial_mag(:,:,kc)=kron(cdata_initial(:,:,kc),mag_kernel_uint8);
    % end
    imwrite(cdata_added,['Masks_MF ',Exp_ID,' ',mat2str(k),'-',num2str(pred_select(ik)),'.png']);
end
