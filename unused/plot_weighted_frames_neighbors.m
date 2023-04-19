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
r_bg_ratio = 2.5;
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
vid=2;
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
[d_diff_overall, d_diff_neighbor_0, d_diff_neighbor, d_diff_out] ...
    = deal(zeros(size(traces_raw),'single'));

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
    list_weight_frame{nn} = min(max_inside-max_outside);
    
    %% Calculate the weight from trace
    dn = d(nn,:);
    if isempty(neighbors)
        d_diff = dn;
    else
        overlap = ROIs2(:,nn)'*ROIs2(:,neighbors);
        consume = overlap./area(neighbors);
        d_neighbors_0 = max(d(neighbors,:),[],1);
        d_neighbors = max(d(neighbors,:)-consume',[],1);
        max_d_neighbors = max(d_out(nn,:),d_neighbors);
        max_d_neighbors = max(0,max_d_neighbors);
        d_diff = dn - max_d_neighbors;
        d_diff_overall(nn,:) = d_diff;
        d_diff_neighbor_0(nn,:) = dn - d_neighbors_0;
        d_diff_neighbor(nn,:) = dn - d_neighbors;
        d_diff_out(nn,:) = dn - d_out(nn,:);
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
    list_weight_trace{nn} = d_diff;
    
    %% Combined weight
    list_weight{nn} = sqrt(list_weight_trace{nn}.*list_weight_frame{nn});
end

diff_frame = cell2mat(list_weight_frame');
%%
bad_out = (d-2).*(-d_diff_out).*(d_diff_neighbor_0>0).*(d_diff_out<0);
bad_out_1 = reshape(bad_out,1,[]);
[bad_out_1_sort,bad_out_1_sort_ind] = sort(bad_out_1,'descend');
%%
bad_neighbor_0 = (d-2).*(-d_diff_neighbor_0).*(d_diff_neighbor_0<0);
nn = 100;
[bad_neighbor_0_sort,bad_neighbor_0_sort_ind] = sort(bad_neighbor_0(nn,:),'descend');
%%
good_neighbor = (d-2).*(d_diff_neighbor).*(d_diff_neighbor_0<0).*(d_diff_neighbor>0);
nn = 213;
[good_neighbor_sort,good_neighbor_sort_ind] = sort(good_neighbor(nn,:),'descend');

%%
t = good_neighbor_sort_ind(1);
% t = bad_neighbor_0_sort_ind(13);
% ind = 51; % 3;
% val = bad_out_1_sort(nn);
% [nn,t] = ind2sub(size(d),bad_out_1_sort_ind(ind));

neighbors = list_neighbors{nn};
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

video_sub_t = video_sub(:,:,t);

figure; 
imagesc(video_sub_t);
colormap gray;
colorbar;
axis('image');
hold on;
contour(mask_sub,'b');
title(sprintf('Neuron %d, frame %d',nn,t))
for kk = 1:length(neighbors)
    contour(FinalMasks(xmin:xmax,ymin:ymax,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
end
image_green = zeros(Lxm,Lym,3,'uint8');
image_green(:,:,2)=255;
edge_others = sum_edges(xmin:xmax,ymin:ymax)-sum(FinalEdges(xmin:xmax,ymin:ymax,[nn,neighbors]),3);
%     edge_others = sum_edges(xmin:xmax,ymin:ymax)-edge(mask_sub);
imagesc(image_green,'alphadata',0.5*edge_others);

% saveas(gcf,sprintf('Neuron %d, frame %d.png',nn,t));