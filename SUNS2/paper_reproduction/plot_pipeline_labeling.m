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
dir_video = 'D:\data_TENASPIS';
eid = 1;
Exp_ID = list_Exp_ID{eid};
alpha = 0.8;

save_folder = '.\plot pipeline\';
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

load(['TENASPIS mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
load(['TENASPIS mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');

dir_original_masks = fullfile(dir_video,'original_masks\GT Masks');
load(fullfile(dir_original_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
Masks_initial = FinalMasks;
% Masks_initial_sum = sum(Masks_initial,3);
[Lx,Ly,N1] = size(FinalMasks);
% edge_GT_Masks = 0*FinalMasks;
% for nn = 1:size(FinalMasks,3)
%     edge_GT_Masks(:,:,nn) = edge(FinalMasks(:,:,nn));
% end
% FinalMasks = permute(FinalMasks,[2,1,3]);
% Masks_initial_sum = sum(edge_GT_Masks,3);
% Masks_initial_sum = sum(FinalMasks,3);

dir_added_masks = fullfile(dir_video,'original_masks\added_blockwise\GT Masks');
load(fullfile(dir_added_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
Masks_added = FinalMasks(:,:,N1+1:end);
% Masks_added_sum = sum(Masks_added,3);
% edge_GT_Masks = 0*FinalMasks;
% for nn = (N1+1):size(FinalMasks,3)
%     edge_GT_Masks(:,:,nn) = edge(FinalMasks(:,:,nn));
% end
% FinalMasks = permute(FinalMasks,[2,1,3]);
% Masks_added_sum = sum(edge_GT_Masks,3);
% Masks_added_sum = sum(FinalMasks(:,:,N1+1:end),3);

dir_GT_masks = fullfile(dir_video,'added_refined_masks\GT Masks');
load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
Masks_refined = FinalMasks;
% Masks_refined_sum = sum(Masks_refined,3);
% edge_GT_Masks = 0*FinalMasks;
% for nn = 1:size(FinalMasks,3)
%     edge_GT_Masks(:,:,nn) = edge(FinalMasks(:,:,nn));
% end
% FinalMasks = permute(FinalMasks,[2,1,3]);
% Masks_refined_sum = sum(edge_GT_Masks,3);
% Masks_refined_sum = sum(FinalMasks,3);

%%
rect1=[400,244,80,80];
mag=1;
mag_crop=4;
mag_kernel_uint8 = ones(mag_crop,mag_crop,'uint8');

rect2=[440,260,35,35];
mag_crop2=8;
mag_kernel2_uint8 = ones(mag_crop2,mag_crop2,'uint8');

%% plot initial and added masks
figure; imshow(SNR_max,[2,14]); % colormap gray;
hold on;
% alphaImg = ones(Lx,Ly).*reshape(magenta,1,1,3);
% image(alphaImg,'Alphadata',alpha*(Masks_initial_sum));  
% contour(Masks_initial_sum,'EdgeColor',magenta,'LineWidth',1); % ,magenta
for n = 1:N1
    contour(Masks_initial(:,:,n), 'EdgeColor',magenta,'LineWidth',0.5);
end
% rectangle('Position',[5,470,16,4],'FaceColor','w','LineStyle','None'); % 20 um scale bar
rectangle('Position',rect1,'EdgeColor',color(5,:),'LineWidth',9);
frame=getframe(gcf);
cdata_initial=frame.cdata;
imwrite(cdata_initial,[save_folder, 'Masks_initial ',Exp_ID,'.png']);

%%
zoom1 = cdata_initial(rect1(2)+1:rect1(2)+rect1(4)-1,rect1(1)+1:rect1(1)+rect1(3)-1,:);
% zoom1(74:77,4:23,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end
imwrite(zoom1_mag,fullfile(save_folder,['cdata_initial ',Exp_ID,' zoom.tif']));

%%
% alphaImg = ones(Lx,Ly).*reshape(color(6,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(Masks_added_sum));  
% contour(Masks_added_sum,'EdgeColor',color(6,:),'LineWidth',1); % ,magenta
for n = 1:size(Masks_added,3)
    contour(Masks_added(:,:,n), 'EdgeColor',color(6,:),'LineWidth',0.5);
end
rectangle('Position',rect1,'EdgeColor',color(5,:),'LineWidth',9);
frame=getframe(gcf);
cdata_added=frame.cdata;
imwrite(cdata_added,[save_folder, 'Masks_added ',Exp_ID,'.png']);
%%
rectangle('Position',rect2,'LineStyle','-.','EdgeColor',green,'LineWidth',2);
frame=getframe(gcf);
cdata_added=frame.cdata;
%%
zoom1 = cdata_added(rect1(2)+1:rect1(2)+rect1(4)-1,rect1(1)+1:rect1(1)+rect1(3)-1,:);
% zoom1(74:77,4:23,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end
imwrite(zoom1_mag,fullfile(save_folder,['Masks_added ',Exp_ID,' zoom.tif']));

%%
zoom2 = cdata_added(rect2(2)+1:rect2(2)+rect2(4)-1,rect2(1)+1:rect2(1)+rect2(3)-1,:);
% zoom2(4:5,12:31,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop2,size(zoom2,2)*mag_crop2,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel2_uint8);
end
imwrite(zoom2_mag,fullfile(save_folder,['Masks_added ',Exp_ID,' zoom2.tif']));

%% plot refined masks
figure; imshow(SNR_max,[2,14]); % colormap gray;
hold on;
% alphaImg = ones(Lx,Ly).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(Masks_refined_sum));  
% contour(Masks_refined_sum,'EdgeColor',color(3,:),'LineWidth',1); % ,magenta
for n = 1:size(Masks_refined,3)
    contour(Masks_refined(:,:,n), 'EdgeColor',color(3,:),'LineWidth',0.5);
end
rectangle('Position',[5,470,16,4],'FaceColor','w','LineStyle','None'); % 20 um scale bar
rectangle('Position',rect1,'EdgeColor',color(5,:),'LineWidth',9);
frame=getframe(gcf);
cdata_refined=frame.cdata;
imwrite(cdata_refined,[save_folder, 'Masks_refined ',Exp_ID,'.png']);
%%
rectangle('Position',rect2,'LineStyle','-.','EdgeColor',green,'LineWidth',2);
frame=getframe(gcf);
cdata_refined=frame.cdata;
%%
zoom1 = cdata_refined(rect1(2)+1:rect1(2)+rect1(4)-1,rect1(1)+1:rect1(1)+rect1(3)-1,:);
zoom1(74:77,4:23,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end
imwrite(zoom1_mag,fullfile(save_folder,['Masks_refined ',Exp_ID,' zoom.tif']));

%%
zoom2 = cdata_refined(rect2(2)+1:rect2(2)+rect2(4)-1,rect2(1)+1:rect2(1)+rect2(3)-1,:);
zoom2(4:5,12:31,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop2,size(zoom2,2)*mag_crop2,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel2_uint8);
end
imwrite(zoom2_mag,fullfile(save_folder,['Masks_refined ',Exp_ID,' zoom2.tif']));
