%% Plot intermediate foreground and background
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\data_CNMFE',data_name);
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
dir_sub = 'complete_TUnCaT\4816[1]th4+BGlayer';
%%
eid = 1;
Exp_ID = list_Exp_ID{eid};
SNR_video= h5read(fullfile(path_name, 'complete_TUnCaT\network_input', [Exp_ID,'.h5']),'/network_input');
kernel = h5read(fullfile(path_name, dir_sub, 'Weights', ['Model_CV',num2str(eid-1),'.h5']),'/conv2d_12/conv2d_12/kernel:0');
kernel = reshape(kernel,[1,1,1,4]);
% kernel = reshape(kernel(end:-1:1),[1,1,1,4]);
load(fullfile(path_name, dir_sub, 'output_masks', [Exp_ID,'_intermediate.mat']));
foreground = sum(foreground.*kernel,4);
background = sum(background.*kernel,4);
foreground = permute(foreground,[3,2,1]);
background = permute(background,[3,2,1]);
combine = (tansig(foreground - background+bias)+1)/2;
prob_map = permute(prob_map,[3,2,1]);
figure; imshow3D(SNR_video); title('foreground');
figure; imshow3D((tansig(foreground)+1)/2); title('foreground');
figure; imshow3D((tansig(background)+1)/2); title('background');
figure; imshow3D(prob_map); title('prob_map');
figure; imshow3D(combine); title('combine');

%%
figure('Position',[20,20,1600,800]);
t = 2998;
subplot(2,3,1)
imagesc(SNR_video(:,:,t),[0,3]); % 
axis('image','off'); colorbar; colormap gray;
title('SNR');

subplot(2,3,4)
imagesc(combine(:,:,t),[0,1]);
axis('image','off'); colorbar; colormap gray;
title('Probability');

subplot(2,3,2)
imagesc(foreground(:,:,t)); % ,[0,3]
axis('image','off'); colorbar; colormap gray;
title('Foreground');

subplot(2,3,5)
imagesc(background(:,:,t)); % ,[0,3]
axis('image','off'); colorbar; colormap gray;
title('Background');

subplot(2,3,3)
imagesc((tansig(foreground(:,:,t))+1)/2 ,[0,1]); %
axis('image','off'); colorbar; colormap gray;
title('Sigmoid Foreground');

subplot(2,3,6)
imagesc((tansig(background(:,:,t))+1)/2 ,[0,1]); %
axis('image','off'); colorbar; colormap gray;
title('Sigmoid Background');

saveas(gcf,['foreground and background t=',num2str(t),'.png']);
