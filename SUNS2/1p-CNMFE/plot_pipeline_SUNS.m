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
dir_video = 'D:\data_TENASPIS\added_refined_masks';
eid = 1;
Exp_ID = list_Exp_ID{eid};
dir_output = fullfile(dir_video,'complete_TUnCaT_SF25\4816[1]th4\output_masks');
load(fullfile(dir_output,['Output_Masks_',Exp_ID,'.mat']),'Masks')
% load('FinalMasks from video part.mat')
FinalMasks=logical(permute(Masks,[3,2,1]));
img = sum(FinalMasks,3)>0;
img = img.*ones(1,1,3);
% figure; imshow(img,[]);
[Lx, Ly, ncells] = size(FinalMasks);
[yy,xx] = meshgrid(1:Lx,1:Ly);
area = squeeze(sum(sum(FinalMasks,1),2));
COMx = squeeze(sum(sum(FinalMasks.*xx,1),2))./area;
COMy = squeeze(sum(sum(FinalMasks.*yy,1),2))./area;
r_neighbor = 30;

save_folder = '.\plot pipeline\';
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

color_1D = 0:1/15:1;
[color1, color2, color3] = ndgrid(color_1D,color_1D,color_1D);
color_all = [color1(:),color2(:),color3(:)];
color_all_sum = sum(color_all,2);
color_all(color_all_sum<1 | color_all_sum>2, :)=[];
ncolor = size(color_all,1);
color_perm = randperm(ncolor);
color_all = color_all(color_perm,:);
color_all_remain = color_all;
color_select = zeros(ncells,3);
c =0;
list_c = zeros(ncells,1);

for k = 1:ncells
    neighbor = find((COMx-COMx(k)).^2+(COMy-COMy(k)).^2<r_neighbor^2);
    neighbor = neighbor(neighbor<k);
    color_neighbor = color_select(neighbor,:);
    color_d = 0;
    while sum(color_d<=0.7)
        c=c+1;
        if c>ncolor
            remain = setdiff(color_perm,list_c);
            ncolor = length(remain);
            color_perm = remain(randperm(ncolor));
            color_all_remain = color_all(color_perm,:);
            list_c = zeros(ncells,1);
            c = 1;
        end
        color_new = color_all_remain(c,:);
        color_d = sum(abs(color_neighbor - color_new),2);
    end
    color_select(k,:)=color_new;
    list_c(k)=c;
end

for k = 1:ncells
    mask = FinalMasks(:,:,k);
    fill=mask.*reshape(color_select(k,:)+0.1,1,1,3);
    img(logical(mask.*ones(1,1,3)))=fill(fill(:)>0)-0.1;
end

% figure; imshow(img);
img_large = 0*padarray(img,[1,1,0]);
img_large(2:end-1,2:end-1,:)=1-img;
figure; imshow(img_large);
% imwrite(img,'FinalMasks_color.png')
imwrite(img_large,[save_folder, 'FinalMasks_color ',Exp_ID,'.png']);
imwrite(1-img_large,[save_folder, 'FinalMasks_color ',Exp_ID,'_black.png']);


%%
% figure; imshow3D(pmap);
% figure; imshow3D(video_raw);
% figure; imshow3D(video_SNR,[0,10]);

%%
load(fullfile(dir_video,'TENASPIS_spike_tempolate.mat'),'filter_tempolate');
tempolate = filter_tempolate(filter_tempolate > exp(-1));
[~,loc] = max(tempolate);
t = 725;
t_raw = t + loc - 1;

%%
dir_video_raw = dir_video;
video_raw = h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),'/mov'); % raw_traces
dir_video_SNR = fullfile(dir_video,'complete_TUnCaT_SF25\network_input\');
video_SNR = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input'); % raw_traces
load(fullfile(dir_output,[Exp_ID,'_pmap.mat']),'prob_map');
pmap = permute(prob_map,[3,2,1]);

%% plot raw frame
img_raw = video_raw(:,:,t_raw);
figure; imshow(img_raw,[]); colormap gray;
frame=getframe(gcf);
cdata_raw=padarray(frame.cdata,[4,4,0],256);
imwrite(cdata_raw,[save_folder, 'Frame_raw ',Exp_ID,' ',num2str(t_raw),'.png']);

%% plot SNR frame
img_SNR = video_SNR(:,:,t);
figure; imshow(img_SNR,[0,10]); colormap gray;
frame=getframe(gcf);
cdata_SNR=padarray(frame.cdata,[4,4,0],256);
imwrite(cdata_SNR,[save_folder, 'Frame_SNR ',Exp_ID,' ',num2str(t),'.png']);

%% plot probability map
img_pmap = pmap(:,:,t);
figure; imshow(img_pmap); colormap gray;
frame=getframe(gcf);
cdata_pmap=padarray(frame.cdata,[4,4,0],256);
imwrite(cdata_pmap,[save_folder, 'Frame_pmap ',Exp_ID,' ',num2str(t),'.png']);

