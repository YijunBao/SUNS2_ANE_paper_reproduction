% global mask_update_select;
% global Lxm;
% global Lym;
% global n_select;
load('crash_data.mat');
n_select = min(n_select,90);
mask_update_select = mask_update_select(:,:,1:n_select);
mask_update_select_2 = reshape(mask_update_select,Lxm*Lym,n_select)';
X = double(mask_update_select_2);
Y =  pdist(X,'jaccard'); 
Y(Y>0.7) = nan;
% video_sub_sort_2 = reshape(video_sub_sort,Lxm*Lym,n_select)';
% X = double(video_sub_sort_2);
% Y =  pdist(X,'cosine'); 
Y2 = squareform(Y);
%% Remove frames with all nan
Y2nan = isnan(Y2);
allnan = (sum(Y2nan,1) == n_select);
mask_update_select = mask_update_select(:,:,~allnan);
mask_update_select_2 = reshape(mask_update_select,Lxm*Lym,n_select)';
n_select = size(mask_update_select,3);
X = double(mask_update_select_2);
Y =  pdist(X,'jaccard'); 
% Y(Y>0.7) = nan;
Y2 = squareform(Y);
%%
% Z = linkage(Y,'single');
Z = linkage(Y,'average');
T = cluster(Z,'Cutoff',0.7, 'Criterion','distance');

figure('Position',get(0,'ScreenSize')); dendrogram(Z, n_select);
figure; imagesc(Y2); colorbar; axis('image');

arrayfun(@(x) find(T==x),1:max(T),'UniformOutput',false)