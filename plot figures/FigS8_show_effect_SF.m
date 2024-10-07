list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
path_name = 'D:/data_TENASPIS/added_refined_masks';
dir_SNR1 = fullfile(path_name,'complete_TUnCaT_noSF\network_input');
dir_SNR2 = fullfile(path_name,'complete_TUnCaT_SF25\network_input');

dFF = h5read('TENASPIS_spike_tempolate.h5','/filter_tempolate');
dFF = dFF(dFF>exp(-1));
% dFF = dFF'/sum(dFF);
[~, delay] = max(dFF);
delay = delay -1;

%%
ii = 6;
Exp_ID = list_Exp_ID{ii};
t = 222;
Lx = 480;
Ly = 480;

%%
video_raw = h5read(fullfile(path_name,[Exp_ID,'.h5']),'/mov',[1,1,t+delay],[Lx,Ly,1]);
video_SF = homo_filt(video_raw, 25);
video_SNR1 = h5read(fullfile(dir_SNR1,[Exp_ID,'.h5']),'/network_input',[1,1,t],[Lx,Ly,1]);
video_SNR2 = h5read(fullfile(dir_SNR2,[Exp_ID,'.h5']),'/network_input',[1,1,t],[Lx,Ly,1]);

%%
SNR_range = [-2,10];

figure(11);
set(gcf,'Position',[50,50,600,500],'Color','w');
imagesc(video_SNR1,SNR_range); % 
axis('image'); colormap gray; 
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','SNR');
set(h,'FontSize',14);
saveas(gcf,sprintf('%s_%d_SNR1.png',Exp_ID,t));

figure(12);
set(gcf,'Position',[950,50,600,500],'Color','w');
imagesc(video_SNR2,SNR_range); % 
axis('image'); colormap gray; 
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','SNR');
set(h,'FontSize',14);
saveas(gcf,sprintf('%s_%d_SNR2.png',Exp_ID,t));

%%

figure(21);
set(gcf,'Position',[250,250,600,500],'Color','w');
imagesc(video_raw,[1000,3000]); % ,SNR_range
axis('image'); colormap gray; 
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','Intensity');
set(h,'FontSize',14);
saveas(gcf,sprintf('%s_%d_Raw.png',Exp_ID,t+delay));
    
figure(22);
set(gcf,'Position',[1150,250,600,500],'Color','w');
imagesc(video_SF,[0.85,1.15]); % ,SNR_range
axis('image'); colormap gray; 
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','Intensity');
set(h,'FontSize',14);
saveas(gcf,sprintf('%s_%d_SF.png',Exp_ID,t+delay));
