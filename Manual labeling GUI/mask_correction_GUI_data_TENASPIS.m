%%
global gui;
global txtFntSz;
global video;
global masks;
global area;
global r_bg_ratio;
global comx;
global comy;
global list_weight;
global list_weight_trace;
global list_weight_frame;
global list_neighbors; 
% global list_mask_update; 
% global masks_update; 
global edges;
global sum_edges;
global traces_raw;
global traces_out_exclude;
global traces_bg_exclude;
% global 

%%
% folder of the GT Masks
dir_parent='D:\data_TENASPIS\original_masks\';
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
rate_hz = 20; % frame rate of each video
r_bg_ratio = 2;

%% Load traces and ROIs
vid=8;
Exp_ID = list_Exp_ID{vid};
dir_masks = fullfile(dir_parent,'added_blockwise\GT Masks');
dir_trace = fullfile(dir_parent,'SNR traces');
dir_refine = fullfile(dir_parent,'refine_from_added');
fs = rate_hz;
if ~ exist(Exp_ID,'dir')
    mkdir(Exp_ID);
end
if ~ exist(dir_refine,'dir')
    mkdir(dir_refine);
end

load(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude','traces_out_exclude');
% traces_in = (traces_raw-traces_bg_exclude);
% [mu, sigma] = SNR_normalization(traces_in,'quantile-based std','median');
% d=(traces_in-mu)./sigma; % SNR trace
% d=traces_in; % SNR trace
d = traces_raw;
d_out = traces_out_exclude;

fname=fullfile(dir_trace,['SNR video ', Exp_ID,'.mat']);
load(fname, 'video_SNR');
max_SNR = max(video_SNR,[],3);
T = size(video_SNR,3);
% tile_size = [ceil(T/10/100),10]; % [9,10];

load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
masks=logical(FinalMasks);
[Lx,Ly,N]=size(masks);
ROIs2 = reshape(masks,Lx*Ly,N);
edges = false(Lx,Ly,N);
for nn=1:N
    edges(:,:,nn) = edge(masks(:,:,nn));
end
sum_edges = sum(edges,3);

%% Calculate neighboring neurons by COM distance
list_neighbors = cell(1,N);
comx=zeros(1,N);
comy=zeros(1,N);
for nn=1:N
    [xxs,yys]=find(masks(:,:,nn)>0);
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

%% Calculate the weight of each frame
[list_weight,list_weight_trace,list_d_diff,list_weight_frame] = deal(cell(1,N));
list_thred = zeros(1,N);
thred_min = 1;
video = video_SNR;

for nn = 1:N
    neighbors = list_neighbors{nn};
    %% Calculate the weight from maximum intensity of each frame
    mask = masks(:,:,nn);
    r_bg = sqrt(mean(area)/pi)*r_bg_ratio;
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
    d_diff_sort = sort(d_diff,'descend');
    if T > 6000
        thred = prctile(d_diff,99);
    else
        thred = d_diff_sort(60);
    end
    if thred<thred_min
%         d_diff_sort = sort(d_diff,'descend');
        thred = min(thred_min,d_diff_sort(10));
    end
    list_thred(nn) = thred;
    list_weight_trace{nn} = max(0,d_diff-thred);
    
    %% Combined weight
    list_weight{nn} = sqrt(list_weight_trace{nn}.*list_weight_frame{nn});
end

%%
save(fullfile(dir_refine,[Exp_ID,'_weights.mat']),'list_weight','list_weight_trace','list_weight_frame',...
    'comx','comy','area','list_neighbors','r_bg_ratio','edges','sum_edges',...
    'traces_raw','traces_out_exclude','video','masks');

%%
% global masks_update;
% global list_added;
% global list_delete;
% global ListStrings;
% masks_update = false(Lx,Ly,N);
% list_added = {};
% list_delete = false(1,N);
% ListStrings = num2cell(1:N);
folder = fullfile(dir_refine,Exp_ID);
GUI_refine(video,folder,masks);
% GUI_refine(video,folder,masks,update_result);

