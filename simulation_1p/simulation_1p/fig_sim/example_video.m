%% GT on Raw video
video_show = neuron.reshape(bg_vary, 1) * sigma_bg_invary/sigma_bg_vary;

v = VideoWriter(['Example low-pass BG.avi']);
v.FrameRate = Fs;
open(v);
% figure('Position',[100,100,680,560],'Color','w');

raw_min = single(min(video_show,[],'all'));
raw_max = single(max(video_show,[],'all'));
raw_range = raw_max - raw_min;

for t = 101:200
    img = single(video_show(:,:,t));
    img = (img - raw_min)/raw_range;
    writeVideo(v,single(img));
end
close(v);

