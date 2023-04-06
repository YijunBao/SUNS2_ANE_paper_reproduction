clear;
list_data_names={'noise1'};
list_patch_dims = [253,316]; 
rate_hz = 10; % frame rate of each video
radius = 6;

data_ind = 1;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\simulation_CNMFE',data_name);
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);

dir_method = 'complete_FISSA';
dir_sub=fullfile(dir_method,'4816[1]th4'); %_2out+BGlayer
dir_SNR=fullfile(dir_method,'network_input'); %_2out+BGlayer
dir_temporal_masks=fullfile(dir_method,'temporal_masks(4)'); %_2out+BGlayer
dir_GT_masks=fullfile(dir_video,'GT Masks');
dir_output_masks=fullfile(dir_video,dir_sub,'output_masks');

nframes = 150;
fps = rate_hz(data_ind);
start=[1,1,300];
count=[Inf,Inf,nframes];
stride=[1,1,1];

Lx=list_patch_dims(data_ind,1); 
Ly=list_patch_dims(data_ind,2);
rangex=1:Ly; 
rangey=1:Lx;
crop_img=[86,64,Ly,Lx];

filter_tempolate = h5read(fullfile(dir_video,[data_name,'_spike_tempolate.h5']),'/filter_tempolate');
filter_tempolate = filter_tempolate(filter_tempolate > exp(-1));
[~, offset] = max(filter_tempolate);
offset = offset -1;

k=1;
Exp_ID = list_Exp_ID{k};
color_range_raw = [0,200];
video_raw = h5read(fullfile(dir_video,[Exp_ID,'.h5']),'/mov',start+[0,0,offset], count, stride);
video_raw_min = min(video_raw,[],3);
video_raw = video_raw - video_raw_min;

load(fullfile(dir_GT_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
load(fullfile(dir_output_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks','times_active');
Masks = permute(Masks,[3,2,1]);

color_range_SNR = [-2,5]; % [0,3];
video_SNR = h5read(fullfile(dir_video,dir_SNR,[Exp_ID,'.h5']),'/network_input',start, count, stride);
temporal_masks = h5read(fullfile(dir_video,dir_temporal_masks,[Exp_ID,'.h5']),'/temporal_masks',start, count, stride);

%% Output on raw video
% border = 255*ones(Lxc,10,3,'uint8');
v = VideoWriter([Exp_ID,' GT and SUNS Masks.avi']);
v.FrameRate = fps;
open(v);
figure('Position',[10,10,800,1000],'Color','w');

for t = 1:nframes
    t_real = start(3) + t - 1;
    image_raw = video_raw(:,:,t);
    image_raw = image_raw(rangey,rangex)';
    image_SNR = video_SNR(:,:,t);
    image_SNR = image_SNR(rangey,rangex)';
    
    mask1 = temporal_masks(:,:,t);
    mask1 = mask1(rangey,rangex)';
    neurons_active = cellfun(@(x) ~isempty(find(x==t_real,1)), times_active);
    mask2 = sum(Masks(:,:,neurons_active),3);
    mask2 = mask2(rangey,rangex)';

    subplot(2,2,1);
    cla;
    imshow(image_raw, color_range_raw);
    hold on;
    contour(mask1,'Color', [0.1,0.9,0.1]);
    title('GT on raw video')

    subplot(2,2,2);
    cla;
    imshow(image_raw, color_range_raw);
    hold on;
    contour(mask2,'Color', [0.9,0.1,0.1]);
    title('SUNS on raw video')

    subplot(2,2,3);
    cla;
    imshow(image_SNR, color_range_SNR);
    hold on;
    contour(mask1,'Color', [0.1,0.9,0.1]);
    title('GT on SNR video')

    subplot(2,2,4);
    cla;
    imshow(image_SNR, color_range_SNR);
    hold on;
    contour(mask2,'Color', [0.9,0.1,0.1]);
    title('SUNS on SNR video')

%     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
%     h=colorbar;
%     set(h,'FontSize',9);
%     set(get(h,'Label'),'String','Raw intensity','FontName','Arial');
%     title('Output on raw video')
%     hold on;
%     contour(mask(rangey,rangex)','Color', [0.9,0.1,0.1]);
    pause(0.001);
    img_all=getframe(gcf); % ,crop_img
    img_notrack=img_all.cdata;
    
%     figure(99); imshow(img_both);
    writeVideo(v,img_notrack);
end
close(v);

