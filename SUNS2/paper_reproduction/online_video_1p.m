clear
%% track on raw video
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};

dir_video = 'E:\1photon-small\';
dir_sub1='complete_TUnCaT\'; %_subtract+BGlayer
dir_sub2='test_CNN\4816[1]th6\'; %_subtract+BGlayer
dir_mask = [dir_video,dir_sub1,dir_sub2];
nframes = 5000;
fps = 20;
t_init = 30 * fps;
merge = 20;
start=[1,1,300];
count=[Inf,Inf,nframes];
stride=[1,1,1];
mag=4;

% color_range_raw = [0,40000];
color_range_raw = [0,10];
Lx=50; Ly=50;
% Lxc=210; Lyc=120;
% rangex=200:(200+210-1); rangey=30:(30+120-1);
rangex=1:Ly; rangey=1:Lx;
% crop_png_1=[155,55,100,150];
% crop_png_2=[255,55,45,150];
% crop_png_3=[300,55,15,150];
crop_img=[86,64,Ly*mag,Lx*mag];

for k=9:-1:1
Exp_ID = list_Exp_ID{k};
% video_raw = h5read([dir_video,Exp_ID,'.h5'],'/mov',start, count, stride);
video_raw = h5read([dir_video,dir_sub1,'network_input\',Exp_ID,'.h5'],'/network_input',start, count, stride);
load([dir_mask,'output_masks online video\Output_Masks_',Exp_ID,'.mat'],'list_Masks_2');
% load([dir_mask,'output_masks online track video\Output_Masks_',Exp_ID,'.mat'],'list_active_old','list_Masks_cons_2D');
% list_active_old = list_active_old';
% list_Masks_cons_2D = list_Masks_cons_2D';

mag_kernel = ones(mag,mag,class(video_raw));
mag_kernel_bool = logical(mag_kernel);

%%
mask = sum(reshape(full(list_Masks_2{1}'),Lx,Ly,[]),3);
% mask_track = sum(reshape(full(list_Masks_cons_2D{1}'),Lx,Ly,[]),3);
% border = 255*ones(Lxc,10,3,'uint8');
v = VideoWriter(['Online Masks ',Exp_ID,'.avi']);
v.FrameRate = fps;
open(v);
figure('Position',[100,100,380,260],'Color','w');

for t = 1:nframes
    image = video_raw(:,:,t);
    
%%    masks from SUNS online
    clf; 
    frame = image(rangey,rangex)';
    frame_mag = kron(frame,mag_kernel);
    imshow(frame_mag, color_range_raw);
    if t > t_init && mod(t-4,fps)==0
        t_mask = floor((t-t_init-4)/fps)+1;
        mask = sum(reshape(full(list_Masks_2{t_mask}'),Lx,Ly,[]),3);
    end
%     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
%     h=colorbar;
%     set(h,'FontSize',9);
%     set(get(h,'Label'),'String','Raw intensity','FontName','Arial');
    title('No tracking')
    hold on;
    temp_mask = mask(rangey,rangex)';
    temp_mask_mag = kron(temp_mask,mag_kernel);
    contour(temp_mask_mag,'Color', [0.9,0.1,0.1]);
    pause(0.001);
    img_all=getframe(gcf,crop_img);
    img_notrack=img_all.cdata;
    
    %% masks from tracking SUNS online
%     clf; 
%     frame = image(rangey,rangex)';
%     frame_mag = kron(frame,mag_kernel);
%     imshow(frame_mag, color_range_raw);
%     mask_track = reshape(full(list_Masks_cons_2D{t}'),Lx,Ly,[]);
% %     set(gcf,'Position',get(gcf,'Position')+[0,0,200,0]);
% %     h=colorbar;
% %     set(get(h,'Label'),'String','Raw intensity','FontName','Arial');
% %     set(h,'FontSize',9);
%     title('Tracking')
%     hold on;
%     temp_mask = sum(mask_track(rangey,rangex,:),3)';
%     temp_mask_mag = kron(temp_mask,mag_kernel);
%     contour(temp_mask_mag,'Color', [0.9,0.1,0.1]);
%     act_mask = sum(mask_track(rangey,rangex,list_active_old{t}'),3)';
%     act_mask_mag = kron(act_mask,mag_kernel);
%     contour(act_mask_mag,'Color', [0.1,0.9,0.1]);
%     pause(0.001);
%     img_all=getframe(gcf,crop_img);
% %     img_all=getframe(gcf,crop_png_1);
%     img_track=img_all.cdata;
%     if t==1
%         img_all=getframe(gcf,crop_png_2);
%         img_colorbar=img_all.cdata;
%         img_all=getframe(gcf,crop_png_3);
%         img_label=img_all.cdata;
%     end
    
%     img_both = cat(2,img_notrack, img_track, img_colorbar, img_label);
%     figure(99); imshow(img_both);
    writeVideo(v,img_notrack);
end
close(v);
end
