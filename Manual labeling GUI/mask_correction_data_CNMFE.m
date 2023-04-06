color=[  
%     0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
%     0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];

%%
% folder of the GT Masks
dir_parent='E:\data_CNMFE\';
% name of the videos
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
rate_hz = [10,15,7.5,5]; % frame rate of each video
         
% before=2; % number of frames before spike peak
% after=4; % number of frames after spike peak
r_bg_ratio = 2;
% doesplot=true;
% list_d=[5:6]; % PFC4_15Hz
% list_d=[2:9]; % two element array showing the minimum and maximum allowed SNR
% num_dff=length(list_d)-1;
% [array_tau_s,array_tau2_s]=deal(nan(length(list_Exp_ID),num_dff));
% spikes_avg_all=nan(length(list_Exp_ID), before+after+1);
% time_frame = -before:after;
% d_min = 6;
% d_max = inf;
% MPP = 1;
% num_d = length(list_d);
% [list_onlyone, list_good_amp] = deal(cell(num_d,1));
% figure(97);
% clf;
% set(gcf,'Position',[100,100,500,400]);
% hold on;

%% Load traces and ROIs
vid=1;
Exp_ID = list_Exp_ID{vid};
dir_masks = fullfile(dir_parent);
dir_trace = fullfile(dir_parent,'SNR traces');
fs = rate_hz(vid);
if ~ exist(Exp_ID,'dir')
    mkdir(Exp_ID);
end

fname=fullfile(dir_trace,['SNR video ', Exp_ID,'.mat']);
load(fname, 'video_SNR');
max_SNR = max(video_SNR,[],3);
T = size(video_SNR,3);
tile_size = [ceil(T/10/100),10]; % [9,10];

load(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude','traces_out_exclude');
% traces_in = (traces_raw-traces_bg_exclude);
% [mu, sigma] = SNR_normalization(traces_in,'quantile-based std','median');
% d=(traces_in-mu)./sigma; % SNR trace
% d=traces_in; % SNR trace
d = traces_raw;
d_out = traces_out_exclude;

load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
FinalMasks=logical(FinalMasks);
[Lx,Ly,N]=size(FinalMasks);
ROIs2 = reshape(FinalMasks,Lx*Ly,N);
FinalEdges = false(Lx,Ly,N);
for nn=1:N
    FinalEdges(:,:,nn) = edge(FinalMasks(:,:,nn));
end
sum_edges = sum(FinalEdges,3);
%% Calculate neighboring neurons (have overlapping pixels);
% list_neighbors = cell(1,N);
% Masks_sum = sum(ROIs2,2);
% for nn=1:N
%     mask = ROIs2(:,nn);
%     mask_else = Masks_sum - mask;
%     overlap = mask & mask_else;
%     Masks_overlap = find(sum(ROIs2.*overlap,1));
%     list_neighbors{nn} = setdiff(Masks_overlap,nn);
% end
%% Calculate neighboring neurons by COM distance
list_neighbors = cell(1,N);
comx=zeros(1,N);
comy=zeros(1,N);
for nn=1:N
    [xxs,yys]=find(FinalMasks(:,:,nn)>0);
    comx(nn)=mean(xxs);
    comy(nn)=mean(yys);
end
area = sum(ROIs2,1);
r_bg=sqrt(mean(area)/pi)*2.5;
% [xx,yy] = meshgrid(Lx,Ly);
for nn=1:N
    neighbors = find((comx(nn)-comx).^2 + (comy(nn)-comy).^2 < r_bg^2);
    list_neighbors{nn} = setdiff(neighbors,nn);
end

%% Calculate the average spike shape and decay time
% list_list_peak_locations = cell(N,num_d);
% for ii = 1:num_d
%     d_min = list_d(ii);
%     [onlyone, good_amp, list_peak_locations] = isolate_transients_d_nomix...
%         (d, ROIs2, d_min, d_max, MPP, before, after);
%     list_onlyone{ii} = onlyone;
%     list_good_amp{ii} = good_amp;
%     list_list_peak_locations(:,ii) = list_peak_locations;
% end
% %     figure(97);
% %     plot(time_frame/fs, spikes_avg_all(vid,:), 'LineWidth',2);
% 
% num_good_amp = cell2mat(cellfun(@(x) sum(x,2), list_good_amp, 'UniformOutput',false)');
% num_onlyone = cell2mat(cellfun(@(x) sum(x,2), list_onlyone, 'UniformOutput',false)');
% num_peaks = cellfun(@length, list_list_peak_locations);

%% Calculate the weight of each frame
[list_weight,list_weight_trace,list_d_diff,list_weight_frame] = deal(cell(1,N));
list_thred = zeros(1,N);
thred_min = 1;
for nn = 1:N
    neighbors = list_neighbors{nn};
    %% Calculate the weight from maximum intensity of each frame
    mask = FinalMasks(:,:,nn);
    r_bg = sqrt(mean(area(nn))/pi)*r_bg_ratio;
    xmin = max(1,round(comx(nn)-r_bg));
    xmax = min(Lx,round(comx(nn)+r_bg));
    ymin = max(1,round(comy(nn)-r_bg));
    ymax = min(Ly,round(comy(nn)+r_bg));
    
    mask_sub = mask(xmin:xmax,ymin:ymax);
    [Lxm,Lym] = size(mask_sub);
    mask_sub_2 = reshape(mask_sub,Lxm*Lym,1);
    video_sub = video_SNR(xmin:xmax,ymin:ymax,:);
    video_sub_2 = reshape(video_sub,Lxm*Lym,T);
    
    [yy,xx] = meshgrid(ymin:ymax,xmin:xmax);
    nearby_disk = ((xx-comx(nn)).^2 + (yy-comy(nn)).^2) < (r_bg)^2;
    nearby_disk_2 = reshape(nearby_disk,Lxm*Lym,1);
    nearby_outside_2 = nearby_disk_2 & ~mask_sub_2;
    
%     max_inside = prctile(video_sub_2(mask_sub_2,:),95,1);
%     max_outside = prctile(video_sub_2(nearby_outside_2,:),95,1);
    order_compare = 3:10;
    sort_inside = sort(video_sub_2(mask_sub_2,:),1,'descend');
    max_inside = sort_inside(order_compare,:);
    sort_outside = sort(video_sub_2(nearby_outside_2,:),1,'descend');
%     sort_outside = sort(video_sub_2(~mask_sub_2,:),1,'descend');
    max_outside = sort_outside(order_compare,:);
    list_weight_frame{nn} = max(0,min(max_inside-max_outside));
    
    %% Calculate the weight from trace
    dn = d(nn,:);
    if isempty(neighbors)
        d_diff = dn;
    else
        overlap = ROIs2(:,nn)'*ROIs2(:,neighbors);
        consume = overlap./area(neighbors);
        d_neighbors = max(d(neighbors,:)-consume',[],1);
        max_d_neighbors = max(d_out(nn,:),d_neighbors);
        max_d_neighbors = max(0,max_d_neighbors);
        d_diff = dn - max_d_neighbors;
    end
    list_d_diff{nn} = d_diff;
    weight_frame = list_weight_frame{nn};
    d_diff(weight_frame<=0) = 0;
    thred = prctile(d_diff,99);
    if thred<thred_min
        d_diff_sort = sort(d_diff,'descend');
        thred = min(thred_min,d_diff_sort(10));
    end
    list_thred(nn) = thred;
    list_weight_trace{nn} = max(0,d_diff-thred);
    
    %% Combined weight
    list_weight{nn} = sqrt(list_weight_trace{nn}.*list_weight_frame{nn});
end

%% Weighted average of firing frames
list_avg_frame = cell(1,N);
list_mask_update = cell(1,N);
FinalMasks_update = false(Lx,Ly,N);
list_IoU = zeros(1,N);
figure(92); 
clf;
set(gcf,'Position',[0,0,1920,1000]);
%%
% for nn = [97   108   191   319   420   473   494   533   555]
% for nn = [44    62    73    85   122   125   138   153   180   188   192   205]
for nn = 1:N % 473:N % [125,139,158,192] % 
    mask = FinalMasks(:,:,nn);
    r_bg = sqrt(mean(area(nn))/pi)*r_bg_ratio;
    r_bg_ext = r_bg*3/2;
    xmin = max(1,round(comx(nn)-r_bg_ext));
    xmax = min(Lx,round(comx(nn)+r_bg_ext));
    ymin = max(1,round(comy(nn)-r_bg_ext));
    ymax = min(Ly,round(comy(nn)+r_bg_ext));
    mask_sub = mask(xmin:xmax,ymin:ymax);
    video_sub = video_SNR(xmin:xmax,ymin:ymax,:);
    [Lxm, Lym] = size(mask_sub);
    
    [yy,xx] = meshgrid(ymin:ymax,xmin:xmax);
    nearby_disk = ((xx-comx(nn)).^2 + (yy-comy(nn)).^2) < (r_bg)^2;
    weight = list_weight{nn};
    select_frames = find(weight>0);
    avg_frame = sum(video_sub(:,:,select_frames).*reshape(weight(select_frames),1,1,[]),3)./sum(weight);
    list_avg_frame{nn} = avg_frame;
    avg_frame_use = avg_frame;
    avg_frame_use(~nearby_disk) = 0;
    thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all'), 'all');
    mask_update = avg_frame_use > thred_inten;
    [L,nL] = bwlabel(mask_update,4);
    if nL>1
        list_area_L = zeros(nL,1);
        for kk = 1:nL
            list_area_L(kk) = sum(L==kk,'all');
        end
        [max_area,iL] = max(list_area_L);
%             for test = 1:3
        while max_area < area(nn)
            thred_inten = thred_inten - 0.1;
%                 thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all')/max_area*area(nn), 'all');
            mask_update = avg_frame_use > thred_inten;
            [L,nL] = bwlabel(mask_update,4);
            list_area_L = zeros(nL,1);
            for kk = 1:nL
                list_area_L(kk) = sum(L==kk,'all');
            end
            [max_area,iL] = max(list_area_L);
        end
        mask_update = (L==iL);
    end

    [L0,nL0] = bwlabel(~mask_update,4);
    if nL0>1
        list_area_L0 = zeros(nL0,1);
        for kk = 1:nL0
            list_area_L0(kk) = sum(L0==kk,'all');
        end
        [max_area,iL0] = max(list_area_L0);
        mask_update = (L0 ~= iL0);
    end
    %%
    area_i = sum(mask_update & mask_sub,'all');
    area_u = sum(mask_update | mask_sub,'all');
    IoU = area_i/area_u;
    list_IoU(nn) = IoU;
    if IoU<0.6 % || length(select_frames)<0.01*T
        disp(nn);
        n_select = length(select_frames);
        [weight_sort,select_frames_order] = sort(weight(select_frames),'descend');
        video_sub_sort = video_sub(:,:,select_frames(select_frames_order));
        images_tile = imtile(video_sub_sort, 'GridSize', tile_size);
        mask_sub_tile = repmat(mask_sub,tile_size);
        q = 1-mean(mask_sub,'all');
        mask_update_select = false(Lxm,Lym,n_select);
        for kk = 1:n_select
            frame = video_sub_sort(:,:,kk);
            frame(~nearby_disk) = 0;
            thred_inten = quantile(frame, q, 'all');
            mask_update_select(:,:,kk) = frame > thred_inten;
        end
%         video_sub_2 = reshape(video_sub(:,:,select_frames(select_frames_order)),Lxm*Lym,n_select);
%         mask_sub_2 = reshape(mask_sub,Lxm*Lym,1);
%         max_inten_each = max(video_sub_2(mask_sub_2,:),[],1);
%         mask_update_select = reshape(video_sub_2>max_inten_each*thred_binary,Lxm,Lym,n_select);
        mask_update_tile = imtile(mask_update_select, 'GridSize', tile_size);
%         mask_update_tile = repmat(mask_update,tile_size);
%         edge_others = sum_edges(xmin:xmax,ymin:ymax)-edge(mask_sub);
%         edge_others_tile = repmat(edge_others,tile_size);
        
        max_inten = max(avg_frame(mask_sub));
        figure(92); 
        clf;
        subplot(1,4,1:3);
        imagesc(images_tile,[-1,max_inten]);
        colormap gray;
        colorbar;
        axis('image');
        hold on;
%         image_green = zeros(size(images_tile,1),size(images_tile,2),3,'uint8');
%         image_green(:,:,2)=255;
%         imagesc(image_green,'alphadata',0.5*edge_others_tile);
        contour(mask_sub_tile,'b');
        contour(mask_update_tile,'r');
        title(['Neuron ',num2str(nn)]);
        
        subplot(2,4,4);
        imagesc(max_SNR(xmin:xmax,ymin:ymax));
        colormap gray;
        colorbar;
        axis('image');
        hold on;
        neighbors = list_neighbors{nn};
        for kk = 1:length(neighbors)
            contour(FinalMasks(xmin:xmax,ymin:ymax,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
        end
        contour(mask_sub,'b');
        contour(mask_update,'r');
        image_green = zeros(Lxm,Lym,3,'uint8');
        image_green(:,:,2)=255;
        edge_others = sum_edges(xmin:xmax,ymin:ymax)-sum(FinalEdges(xmin:xmax,ymin:ymax,[nn,neighbors]),3);
        imagesc(image_green,'alphadata',0.5*edge_others);
        title('Peak SNR (entire video)')

        subplot(2,4,8);
        imagesc(avg_frame);
        colormap gray;
        colorbar;
        axis('image');
        hold on;
        neighbors = list_neighbors{nn};
        for kk = 1:length(neighbors)
            contour(FinalMasks(xmin:xmax,ymin:ymax,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
        end
        contour(mask_sub,'b');
        contour(mask_update,'r');
        image_green = zeros(Lxm,Lym,3,'uint8');
        image_green(:,:,2)=255;
        edge_others = sum_edges(xmin:xmax,ymin:ymax)-sum(FinalEdges(xmin:xmax,ymin:ymax,[nn,neighbors]),3);
        imagesc(image_green,'alphadata',0.5*edge_others);
        title(sprintf('Average frame, IoU = %0.2f',IoU))
%         pause;
        saveas(gcf,fullfile(Exp_ID,[num2str(nn),' selected frames.png']));
    end
    %%
    list_mask_update{nn} = mask_update;
%         min_inten = min(avg_frame(mask_sub));
%         mask_update = avg_frame > 0.2*(max_inten-min_inten)+min_inten;
    FinalMasks_update(xmin:xmax,ymin:ymax,nn) = mask_update;

    figure(91);
    set(gcf,'Position',[100,100,400,300]);
    clf;
    imagesc(avg_frame);
    colormap gray;
    colorbar;
    axis('image');
    hold on;
    contour(mask_sub,'b');
    contour(mask_update,'r');
    title(sprintf('Neuron %d, IoU = %0.2f',nn,IoU))
    saveas(gcf,fullfile(Exp_ID,[num2str(nn),'.png']));

%     clf;
%     imagesc(avg_frame);
%     colormap gray;
%     colorbar;
%     axis('image');
%     hold on;
    neighbors = list_neighbors{nn};
    for kk = 1:length(neighbors)
        contour(FinalMasks(xmin:xmax,ymin:ymax,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
    end
%     contour(mask_sub,'b');
%     contour(mask_update,'r');
    image_green = zeros(Lxm,Lym,3,'uint8');
    image_green(:,:,2)=255;
    edge_others = sum_edges(xmin:xmax,ymin:ymax)-sum(FinalEdges(xmin:xmax,ymin:ymax,[nn,neighbors]),3);
%     edge_others = sum_edges(xmin:xmax,ymin:ymax)-edge(mask_sub);
    imagesc(image_green,'alphadata',0.5*edge_others);
%     contour(sum_edges(xmin:xmax,ymin:ymax)-edge(mask_sub),'y');
    title(sprintf('Neuron %d, IoU = %0.2f',nn,IoU))
    saveas(gcf,fullfile(Exp_ID,[num2str(nn),' and others.png']));
end
save(['FinalMasks_',Exp_ID,'_update.mat'],'FinalMasks_update','list_IoU')


