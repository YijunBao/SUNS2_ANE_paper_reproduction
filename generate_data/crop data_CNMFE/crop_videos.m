%%
clear;
% dir_parent='E:\data_CNMFE';
dir_parent='.';
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);

xyrange = [ 1, 120, 137, 256, 1, 120, 137, 256;
            1,  80,  96, 175, 1,  80, 105, 184;
            1,  88, 113, 200, 1,  88, 113, 200;
            1, 192, 210, 401, 1, 240, 269, 508]; % lateral dimensions to crop four sub-videos.

%%
for ind=1:4 % 1 % 
    data_name = list_Exp_ID{ind};
    dir_data = fullfile(dir_parent, data_name);
    if ~exist(fullfile(dir_parent,data_name,'GT Masks'),'dir')
        mkdir(fullfile(dir_parent,data_name,'GT Masks'))
    end

    %% load movies in the mov h5 file
    load(fullfile(dir_parent,[data_name,'.mat']),'Y','Ysiz');
    img = Y;
    [w, h, t] = size(img);
    type = class(img);
    for xpart = 1:2
        for ypart = 1:2
            xrange = xyrange(ind,2*xpart-1):xyrange(ind,2*xpart);
            yrange = xyrange(ind,2*ypart-1+4):xyrange(ind,2*ypart+4);
            mov = img(xrange,yrange,:);

            h5_name = fullfile(dir_parent,data_name,sprintf('%s_part%d%d.h5',data_name,xpart,ypart));
            if exist(h5_name,'file')
                delete(h5_name)
            end
            h5create(h5_name,'/mov',size(mov),'Datatype',type);
            h5write(h5_name,'/mov',mov);
        end
    end
    
    %% load the GT masks (training data only)
    load(fullfile(dir_parent,['GT Masks updated/FinalMasks_',data_name,'_merge.mat']),'FinalMasks');
    masks = logical(FinalMasks);
    areas = squeeze(sum(sum(masks,1),2));
    avg_area = median(areas)
    
    for xpart = 1:2
        for ypart = 1:2
            xrange = xyrange(ind,2*xpart-1):xyrange(ind,2*xpart);
            yrange = xyrange(ind,2*ypart-1+4):xyrange(ind,2*ypart+4);
            FinalMasks = masks(xrange,yrange,:);
            areas_cut = squeeze(sum(sum(FinalMasks,1),2));
            areas_ratio = areas_cut./areas;
            FinalMasks(:,:,areas_ratio<1/3)=[];
            mask_name = fullfile(dir_parent,data_name,'GT Masks',sprintf('FinalMasks_%s_part%d%d.mat',data_name,xpart,ypart));
            save(mask_name,'FinalMasks','-v7.3');
        end
    end
end