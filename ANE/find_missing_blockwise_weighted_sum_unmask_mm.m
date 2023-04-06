function [image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss, select_frames_id] = ...
    find_missing_blockwise_weighted_sum_unmask_mm(mm, masks, xmin, xmax, ymin, ymax, ...
    weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split)
% T=mm.Format{2}(3);
% [Lx,Ly,T] = size(video);
[Lx,Ly,N] = size(masks);
%% Update interface
% mask = masks(:,:,nn);
% r_bg_ext = round(r_bg*(r_bg_ratio+1)/r_bg_ratio);
% Leng = 2*r_bg_ext+1;
% xmin = max(1,round(comx(nn))-r_bg_ext);
% xmax = min(Lx,round(comx(nn))+r_bg_ext);
% ymin = max(1,round(comy(nn))-r_bg_ext);
% ymax = min(Ly,round(comy(nn))+r_bg_ext);
xrange = xmin:xmax;
yrange = ymin:ymax;
% masks_sum_sub = masks_sum(xrange,yrange);
video_sub = mm.Data.video(xrange,yrange,:);
masks_sub = masks(xmin:xmax,ymin:ymax,:);
neighbors = squeeze(sum(sum(masks_sub,1),2)) > 0;
masks_neighbors = masks_sub(:,:,neighbors);
[Lxm,Lym,num_neighbors] = size(masks_neighbors);
unmasked = ~sum(masks_neighbors,3);
q = 1-avg_area/(Lxm*Lym);

% [yy,xx] = meshgrid(yrange,xrange);
% nearby_disk = ((xx-comx(nn)).^2 + (yy-comy(nn)).^2) < (r_bg)^2;
% weight = list_weight{nn};
near_zone = false(Lxm,Lym);
% if xmin == 1
%     xmin_near = 1;
% else
%     xmin_near = round(Lxm/4+1);
% end
% if xmax == Lx
%     xmax_near = Lxm;
% else
%     xmax_near = round(Lxm*3/4);
% end
% if ymin == 1
%     ymin_near = 1;
% else
%     ymin_near = round(Lym/4+1);
% end
% if ymax == Ly
%     ymax_near = Lym;
% else
%     ymax_near = round(Lym*3/4);
% end
% near_zone(xmin_near:xmax_near, ymin_near:ymax_near) = true;
near_zone(round(Lxm/4+1):round(Lxm*3/4), round(Lym/4+1):round(Lym*3/4)) = true;
far_zone = ~near_zone;
near_zone_2 = reshape(near_zone, Lxm*Lym,1);
far_zone_2 = reshape(far_zone, Lxm*Lym,1);

select_frames = find(weight>0);
[weight_sort,select_frames_order] = sort(weight(select_frames),'descend');
select_frames_sort = select_frames(select_frames_order);

% Exclude frames that do not have qualified binary masks
list_good = zeros(1,num_avg);
g = 0;
for t = 1:length(select_frames_sort)
    t_frame = select_frames_sort(t);
    frame = video_sub(:,:,t_frame);
%     frame(~nearby_disk) = 0;
    thred_inten = quantile(frame, q, 'all');
    [frame_thred, noisy] = threshold_frame(frame, thred_inten, avg_area, unmasked);
    if ~noisy
        nearby = sum(frame_thred & far_zone, 'all') < sum(frame_thred & near_zone, 'all')*2;
        if nearby % Otherwise, better to be found when dealing with another neuron.
            g = g + 1;
            list_good(g) = t_frame;
            if g >= num_avg
                break;
            end
        end
    end
end

if g >= num_avg
    select_frames_sort = list_good;
%     select_frames_order = 1:num_avg;
else
    select_frames_sort = list_good(1:g);
%     select_frames_order = 1:g;
end

%% Update tiles
n_select = length(select_frames_sort);
video_sub_sort = video_sub(:,:,select_frames_sort);
mask_update_select = false(Lxm,Lym,n_select);
list_noisy = false(1,n_select);
for kk = 1:n_select
    frame = video_sub_sort(:,:,kk);
%     frame(~nearby_disk) = 0;
    thred_inten = quantile(frame, q, 'all');
    [frame_thred, noisy] = threshold_frame(frame, thred_inten, avg_area, unmasked);
    mask_update_select(:,:,kk) = frame_thred;
    list_noisy(kk) = noisy;
end

%% Clustering
select_frames_sort_full = select_frames_sort;
if n_select == 0
    classes = [];
    list_far = false(1,0);
    list_class_frames = {};
elseif n_select == 1
    classes = 0;
    list_far = false;
    list_class_frames = {};
else
    mask_update_select_2 = reshape(mask_update_select,Lxm*Lym,n_select)';
    mask_update_select_2(list_noisy,:) = 0;
    dist =  pdist(double(mask_update_select_2),'jaccard'); 
%     dist(dist>thred_jaccard) = nan;
    tree = linkage(dist,'average');
%         figure(88); dendrogram(tree, n_select); ylim([0,thred_jaccard]);
    classes = cluster(tree, 'Cutoff',thj, 'Criterion','distance');
%     classes = clusterdata(double(mask_update_select_2),'MaxClust',2,'Distance','jaccard','Linkage','average');
%     video_sub_sort_2 = reshape(video_sub_sort,Lxm*Lym,n_select)';
%     classes = clusterdata(video_sub_sort_2,'Distance','cosine','MaxClust',2);
%     if sum(classes == 1) < sum(classes == 2)
%         classes = 3 - classes;
%     end
%     classes = ones(1,n_select);

    dist_2 = squareform(dist);
    n_class = max(classes);
    valid_cluster = false(1,n_class);
    list_classes = arrayfun(@(x) find(classes==x),1:max(classes),'UniformOutput',false);
    list_avg_mask_2 = cell(1,n_class);
%     mask_sub_2 = reshape(mask_sub,1,Lxm*Lym);
%     masks_sum_sub_2 = reshape(masks_sum_sub,1,Lxm*Lym);
    list_far = false(1,n_select);
    for c = 1:n_class
        class_frames = list_classes{c};
        dist_2_c = dist_2(class_frames,class_frames) + eye(length(class_frames));
        min_dist_2_c = min(dist_2_c,[],'all');
        if min_dist_2_c < thj_inclass
            valid_cluster(c) = true;
            avg_mask_2 = sum(mask_update_select_2(class_frames,:),1)';
            avg_mask_2 = avg_mask_2 > 0.5*max(avg_mask_2);
            list_avg_mask_2{c} = avg_mask_2;

%             % Remove potential missing neurons that are much closer to other neurons
            nearby = sum(avg_mask_2 & far_zone_2, 'all') < sum(avg_mask_2 & near_zone_2, 'all')*2;
            if ~nearby
                valid_cluster(c) = false;
                list_far(class_frames) = true;
            end
        end
    end
    avg_mask_2_all = cell2mat(list_avg_mask_2(valid_cluster));
    list_classes_valid = list_classes(valid_cluster);
    [avg_mask_2_merge,list_class_frames] = piece_neurons_IOU(avg_mask_2_all,0.5,th_IoU_split,list_classes_valid);
end

%% Multiple neurons
n_class = length(list_class_frames);
classes = zeros(1,n_select);
select_frames_sort = cell(1,n_class);
for c = 1:n_class
    classes(list_class_frames{c}) = c;
    select_frames_sort{c} = select_frames_sort_full(list_class_frames{c});
end
classes(list_far) = -1;

[avg_frame,mask_update,select_frames_class,select_weight_calss] = deal(cell(1,n_class));
for c = 1:n_class
    select_frames_class{c} = video_sub(:,:,select_frames_sort{c});
    select_weight_calss{c} = weight(select_frames_sort{c});
%     avg_frame{c} = mean(select_frames_class{c},3);
    avg_frame{c} = sum(select_frames_class{c}.*reshape(select_weight_calss{c},1,1,[]),3)...
        /sum(select_weight_calss{c});
    avg_frame_use = avg_frame{c};
%     avg_frame_use(~nearby_disk) = 0;
    thred_inten = quantile(avg_frame_use, q, 'all');
    mask_update{c} = threshold_frame(avg_frame_use, thred_inten, avg_area, unmasked);
end

%% confirm select multi
%     list_valid = false(1,n_class);
mask_new_full = false(Lx,Ly,n_class);
mask_new_crop = false(Lxm,Lym,n_class);
image_new_crop = zeros(Lxm,Lym,n_class,'single');
for c = 1:n_class
%         list_valid(c) = true; % get(gui.CheckBox{c},'Value');
    mask_new_full(xrange,yrange,c) = mask_update{c};
    mask_new_crop(:,:,c) = mask_update{c};
    image_new_crop(:,:,c) = avg_frame{c};
end

