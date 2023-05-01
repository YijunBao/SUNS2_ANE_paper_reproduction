%%
clear;
% dir_parent='D:/data_TENASPIS';
dir_parent='.';
% Names of the motion corrected full videos in mat format
list_Exp_ID_read={'Mouse_1K recording_20141122_083913-002_normcorr',...
    'Mouse_2K recording_20141218_095135-002_normcorr',...
    'Mouse_3K recording_20150831_083050-002_normcorr',...
    'Mouse_4K recording_20150901_103340-002_normcorr',...
    'Mouse_1M recording_20151202_172313-006_normcorr',...
    'Mouse_2M recording_20160421_125208-007_normcorr',...
    'Mouse_3M recording_20160705_181552-006_normcorr',...
    'Mouse_4M recording_20160708_172756-007_normcorr'};
% Names of the cropped videos saved in h5 format
list_Exp_ID_save = {'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
                    'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID_read);
sub = 'cropped videos';

xyrange = [ 31,  510, 121, 600;
            31,  510, 121, 600;
            31,  510, 211, 690;
            31,  510, 121, 600;
            31,  510, 211, 690;
            31,  510, 121, 600;
            31,  510,  61, 540;
            31,  510,  31, 510]; % lateral dimensions to crop four sub-videos.
trange = {1:2061, 201:1793,   1:1792, 201:2000, ...
        201:2000,   1:2061, 301:2000, 301:2200};

%%
for ind=1:num_Exp
    data_name = list_Exp_ID_read{ind};
    Exp_ID_save = list_Exp_ID_save{ind};
    if ~exist(fullfile(dir_parent,sub),'dir')
        mkdir(fullfile(dir_parent,sub))
    end

    %% load movies in the mov h5 file
    load(fullfile(dir_parent,[data_name,'.mat']),'Mpr');
    img = Mpr;
    [w, h, t] = size(img);
    type = class(img);
    xrange = xyrange(ind,1):xyrange(ind,2);
    yrange = xyrange(ind,3):xyrange(ind,4);
    mov = img(xrange,yrange,trange{ind});

    h5_name = fullfile(dir_parent,sub,[Exp_ID_save,'.h5']);
    if exist(h5_name,'file')
        delete(h5_name)
    end
    h5create(h5_name,'/mov',size(mov),'Datatype',type);
    h5write(h5_name,'/mov',mov);
end