list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
dir_video = 'D:\data_TENASPIS\added_refined_masks';
eid = 1;
Exp_ID = list_Exp_ID{eid};
load(fullfile(dir_video,'complete_TUnCaT_SF25\Mouse_1K_TF.mat'))
load(fullfile(dir_video,'complete_TUnCaT_SF25\Mouse_1K_SF.mat'))
video_SF = permute(video_SF,[3,2,1]);
video_TF = permute(video_TF,[3,2,1]);

%%
load(fullfile(dir_video,'TENASPIS_spike_tempolate.mat'),'filter_tempolate');
tempolate = filter_tempolate(filter_tempolate > exp(-1));
[~,loc] = max(tempolate);
t = 725;
t_raw = t + loc - 1;
save_folder = '.\';
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

%% plot SF frame
img_SF = video_SF(:,:,t_raw);
figure; imshow(img_SF,[],'border','tight'); colormap gray;
frame=getframe(gcf);
cdata_raw=padarray(frame.cdata,[4,4,0],256);
imwrite(cdata_raw,[save_folder, 'Frame_SF ',Exp_ID,' ',num2str(t_raw),'.png']);

%% plot TF frame
img_TF = video_TF(:,:,t);
figure; imshow(img_TF,[],'border','tight'); colormap gray;
frame=getframe(gcf);
cdata_SNR=padarray(frame.cdata,[4,4,0],256);
imwrite(cdata_SNR,[save_folder, 'Frame_TF ',Exp_ID,' ',num2str(t),'.png']);

