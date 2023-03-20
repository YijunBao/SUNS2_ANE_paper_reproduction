function [list_weight,list_weight_trace,list_weight_frame, sum_edges]...
    = frame_weight_blockwise(video, d, masks, leng)

[Lx,Ly,T]=size(video);
N = size(masks,3);
npatchx = ceil(Lx/leng)-1;
npatchy = ceil(Ly/leng)-1;
% ROIs2 = reshape(masks,Lx*Ly,N);
edges = false(Lx,Ly,N);
for nn=1:N
    edges(:,:,nn) = edge(masks(:,:,nn));
end
sum_edges = sum(edges,3);

%% Calculate the weight of each frame
[list_weight,list_weight_trace,list_d_diff,list_weight_frame] = deal(cell(npatchx,npatchy));

for ix = 1:npatchx
for iy = 1:npatchy
%     disp([ix,iy]);
    %% Calculate the weight from maximum intensity of each frame
%     mask = masks(:,:,nn);
%     r_bg = sqrt(mean(area)/pi)*r_bg_ratio;
    xmin = min(Lx-2*leng+1, (ix-1)*leng+1);
    xmax = min(Lx, (ix+1)*leng);
    ymin = min(Ly-2*leng+1, (iy-1)*leng+1);
    ymax = min(Ly, (iy+1)*leng);
    
    masks_sub = masks(xmin:xmax,ymin:ymax,:);
    neighbors = squeeze(sum(sum(masks_sub,1),2)) > 0;
    masks_neighbors = masks_sub(:,:,neighbors);
    [Lxm,Lym,num_neighbors] = size(masks_neighbors);
%     masks_neighbors_2 = reshape(masks_neighbors,Lxm*Lym,num_neighbors);
    union_neighbors = sum(masks_neighbors,3)>0;
    union_neighbors_2 = reshape(union_neighbors,Lxm*Lym,1);
    video_sub = video(xmin:xmax,ymin:ymax,:);
    video_sub_2 = reshape(video_sub,Lxm*Lym,T);
    
%     [yy,xx] = meshgrid(ymin:ymax,xmin:xmax);
%     nearby_disk = ((xx-comx(nn)).^2 + (yy-comy(nn)).^2) < (r_bg)^2;
%     nearby_disk_2 = reshape(nearby_disk,Lxm*Lym,1);
    nearby_outside_2 = ~union_neighbors_2;
    
%     max_inside = prctile(video_sub_2(mask_sub_2,:),95,1);
%     max_outside = prctile(video_sub_2(nearby_outside_2,:),95,1);
    order_compare = 3:10;
    if min(sum(union_neighbors_2), sum(nearby_outside_2)) < max(order_compare)
        list_weight_frame{ix,iy} = 0;
    else
        sort_inside = sort(video_sub_2(union_neighbors_2,:),1,'descend');
        max_inside = sort_inside(order_compare,:);
        sort_outside = sort(video_sub_2(nearby_outside_2,:),1,'descend');
    %     sort_outside = sort(video_sub_2(~mask_sub_2,:),1,'descend');
        max_outside = sort_outside(order_compare,:);
        list_weight_frame{ix,iy} = max(0,min(max_outside-max_inside));
    end
    
    %% Calculate the weight from trace
    d_out = mean(video_sub_2(nearby_outside_2,:),1);
    d_neurons_max = max(d(neighbors,:),[],1);
    d_diff = d_out - d_neurons_max;
    list_d_diff{ix,iy} = d_diff;
    list_weight_trace{ix,iy} = max(0,d_diff);
    
    %% Combined weight
    list_weight{ix,iy} = max(list_weight_trace{ix,iy},list_weight_frame{ix,iy});
%     list_weight{ix,iy} = sqrt(list_weight_trace{ix,iy}.*list_weight_frame{ix,iy});
end
end