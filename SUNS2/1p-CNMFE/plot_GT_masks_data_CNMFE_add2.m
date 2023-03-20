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

save_figures = true;
mag=4;
mag_kernel = ones(mag,mag,'uint8');
SNR_range = [2,10];
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% neurons and masks frame
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_title = {'upper left', 'upper right', 'lower left', 'lower right'};
sub_added = 'add_neurons_0.001_rotate'; % _lowoverlap

data_ind = 1;
data_name = list_data_names{data_ind};
dir_original = fullfile('E:\data_CNMFE',[data_name,'_original_masks']);
dir_video = fullfile('E:\data_CNMFE',[data_name,'_added_blockwise_weighted_sum_unmask'],sub_added);
dir_video_raw = dir_video;
% varname = '/mov';
dir_video_SNR = fullfile(dir_video,'complete_TUnCaT\network_input\');
list_Exp_ID = cellfun(@(x) [data_name,x,'_added'], list_ID_part,'UniformOutput',false);
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_GT_original = fullfile(dir_original,'GT Masks');
dir_GT_masks = fullfile(dir_video,'GT Masks');
dir_GT_info = fullfile(dir_video,'GT info'); % FinalMasks_

for k=1:4
    % clear video_SNR video_raw
    Exp_ID = list_Exp_ID{k};
    % load(['ABO mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
    % SNR_max=SNR_max';
    % load(['ABO mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');
    % raw_max=raw_max';
    % unmixed_traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
    video_raw = h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),'/mov'); % raw_traces
    raw_max = max(video_raw,[],3);
    video_SNR = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input'); % raw_traces
    SNR_max = max(video_SNR,[],3);
    [Lx,Ly] = size(raw_max);
    SNR_max = SNR_max(1:Lx,1:Ly);

    load(fullfile(dir_GT_info,['GT_',Exp_ID,'.mat']),'masks','masks_add','Ysiz');
    masks_add2 = masks_add;
    load(fullfile(dir_GT_original, ['\FinalMasks_', Exp_ID(1:end-6), '.mat']), 'FinalMasks');
    masks_original = FinalMasks;
    masks_add1 = masks(:,:,size(FinalMasks,3)+1:end);
    
    edge_masks_original = 0*masks_original;
    for nn = 1:size(masks_original,3)
        edge_masks_original(:,:,nn) = edge(masks_original(:,:,nn));
    end
    % FinalMasks = permute(FinalMasks,[2,1,3]);
    masks_original_sum = sum(masks_original,3);
    edge_masks_original_sum = sum(edge_masks_original,3);
    
    edge_masks_add1 = 0*masks_add1;
    for nn = 1:size(masks_add1,3)
        edge_masks_add1(:,:,nn) = edge(masks_add1(:,:,nn));
    end
    % FinalMasks = permute(FinalMasks,[2,1,3]);
    masks_add1_sum = sum(masks_add1,3);
    edge_masks_add1_sum = sum(edge_masks_add1,3);
    
    edge_masks_add2 = 0*masks_add2;
    for nn = 1:size(masks_add2,3)
        edge_masks_add2(:,:,nn) = edge(masks_add2(:,:,nn));
    end
    % FinalMasks = permute(FinalMasks,[2,1,3]);
    masks_add2_sum = sum(masks_add2,3);
    edge_masks_add2_sum = sum(edge_masks_add2,3);

    %% magnify
    mag=4;
    mag_kernel = ones(mag,mag,class(SNR_max));
    mag_kernel_bool = logical(mag_kernel);
    SNR_max_mag = kron(SNR_max,mag_kernel);
    [Lxm,Lym] = size(SNR_max_mag);
    masks_original_sum_mag=kron(masks_original_sum,mag_kernel_bool);
    edge_masks_original_sum_mag=kron(edge_masks_original_sum,mag_kernel_bool);
    masks_sum_add1_mag=kron(masks_add1_sum,mag_kernel_bool);
    edge_masks_add1_sum_mag=kron(edge_masks_add1_sum,mag_kernel_bool);
    masks_sum_add2_mag=kron(masks_add2_sum,mag_kernel_bool);
    edge_masks_add2_sum_mag=kron(edge_masks_add2_sum,mag_kernel_bool);

    xrange=1:Lxm; yrange=1:Lym;
    % xrange = 334:487; yrange=132:487;
    % xrange = 1:487; yrange=1:487;
    crop_png=[86,64,length(yrange),length(xrange)];

    %% 
    % Style 2: Three colors
    % figure(98)
    % subplot(2,3,1)
    figure('Position',[50,50,400,300],'Color','w');
    %     imshow(raw_max,[0,1024]);
    imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
    xticklabels({}); yticklabels({});
    hold on;
    alphaImg_original = ones(Lx*mag,Ly*mag).*reshape(yellow,1,1,3);
    alphaImg_add1 = ones(Lx*mag,Ly*mag).*reshape(blue,1,1,3);
    alphaImg_add2 = ones(Lx*mag,Ly*mag).*reshape(magenta,1,1,3);
    alpha = 0.8;
    % contour(masks_TP(xrange,yrange), 'Color', green);
    % contour(masks_FP(xrange,yrange), 'Color', red);
    % contour(masks_FN(xrange,yrange), 'Color', blue);
%     contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
    image(alphaImg_original,'Alphadata',alpha*(edge_masks_original_sum_mag(xrange,yrange)));  
    image(alphaImg_add1,'Alphadata',alpha*(edge_masks_add1_sum_mag(xrange,yrange)));  
    image(alphaImg_add2,'Alphadata',alpha*(edge_masks_add2_sum_mag(xrange,yrange)));  
%     imshow(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
    % contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));

    % title(sprintf('SUNS, F1 = %1.2f',F1),'FontSize',12);
    title(list_title{k});
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);

    % % rectangle('Position',rect1,'EdgeColor',yellow,'LineWidth',2);
    % % rectangle('Position',rect2,'EdgeColor',yellow,'LineWidth',2);
    % rectangle('Position',rect3,'EdgeColor',color(7,:),'LineWidth',2);
    % % rectangle('Position',rect4,'EdgeColor',yellow,'LineWidth',2);
    % rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
    rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

    if save_figures
        saveas(gcf,[data_name, '_added_blockwise_weighted_sum_unmask ', sub_added, ' masks ',list_title{k},'.png']);
        % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
    end

    %% Save tif images with zoomed regions
    % % figure(1)
    % img_all=getframe(gcf,crop_png);
    % cdata=img_all.cdata;
    % % cdata=permute(cdata,[2,1,3]);
    % % figure; imshow(cdata);
    % 
    % if save_figures
    %     imwrite(permute(cdata,[2,1,3]),'Fig3A SUNS.tif');
    % end
end