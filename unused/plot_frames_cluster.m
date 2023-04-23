% Run "pre_CNN_data_CNMFE_crop_GT_blockwise_weighted_sum_Copy.m" first
% load('E:\data_CNMFE\PFC4_15Hz_original_masks\GT Masks\add_new_blockwise_weighted_sum_unmask\PFC4_15Hz_part11_weights_blockwise.mat')
% load('E:\data_CNMFE\PFC4_15Hz_original_masks\GT Masks\add_new_blockwise_weighted_sum_unmask\PFC4_15Hz_part11_added_auto_blockwise.mat')
addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');

data_ind = 2;
list_avg_radius = [5,6,0,0];
r_bg_ratio = 3;
leng = r_bg_ratio*list_avg_radius(data_ind);
th_IoU_split = 0.5;
thj_inclass = 0.4;
thj = 0.7;

[Lx,Ly,T] = size(video);
% [Lx,Ly,N] = size(masks);
ncol = 10;
num_avg = min(90, ceil(T*0.01));
nrow = ceil(num_avg/ncol);
tile_size = [nrow, ncol];
area = squeeze(sum(sum(masks,1),2));
avg_area = mean(area);

%%
ix = 3; 
iy = 2; 
weight = list_weight{ix,iy};
% added_frames = list_added_frames{ix,iy};
% added_crop = list_added_crop{ix,iy};
% added_weights = list_added_weights{ix,iy};
% added_images_crop = list_added_images_crop{ix,iy};
[image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss] = ...
    find_missing_blockwise_plot(video, masks, ix, iy, leng, ...
    weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split);

%%
max3 = max(video,[],3);
figure;
imagesc(max3);
axis('image');
colormap gray;
colorbar;
hold on;
for n = 1:size(masks,3)
    mask = masks(:,:,n);
    contour(mask,'r');
end
xmin = min(Lx-2*leng+1, (ix-1)*leng+1);
xmax = min(Lx, (ix+1)*leng);
ymin = min(Ly-2*leng+1, (iy-1)*leng+1);
ymax = min(Ly, (iy+1)*leng);
rectangle('Position',[ymin,xmin,ymax-ymin+1,xmax-xmin+1], 'EdgeColor','y','LineWidth',2)
saveas(gcf, sprintf('Max image crop (%d,%d).png',ix,iy));
