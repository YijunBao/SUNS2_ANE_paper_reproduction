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

%%
dir_parent='E:\data_CNMFE\';
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_titles={'dorsal striatum','PFC','Ventral hippocampus','BNST'};
dir_raw_max = 'C:\Matlab Files\SUNS-1p\1p-CNMFE\CNMFE_full mat\raw_max\';
dir_SNR_max = 'C:\Matlab Files\SUNS-1p\1p-CNMFE\CNMFE_full mat\SNR_max\';
% dir_data = 'F:\CaImAn data\WEBSITE\max projections\';
dir_Masks = 'E:\data_CNMFE\full videos\GT Masks\';
Dimens = [120, 120;  80,  80; 88, 88; 192, 240];
xyrange = [ 1, 120, 137, 256, 1, 120, 137, 256;
            1,  80,  96, 175, 1,  80, 105, 184;
            1,  88, 113, 200, 1,  88, 113, 200;
            1, 192, 210, 401, 1, 240, 269, 508]; % lateral dimensions to crop four sub-videos.
use_SNR = true; % false; % 
dir_save = 'show crop';
if ~exist(dir_save,'dir')
    mkdir(dir_save)
end
if use_SNR
    color_range = [2,14; 2,14; 2,14; 2,14];
else
    color_range = [1000,3500; 00,3000; 5000,30000; 00,2500];
end
fontsize = 20;

        
%% find tiff files and order them
for ind= 1:4 % 4 % 
    data_name = list_data_names{ind};
%     load([dir_data,'Max projection',data_name,'.mat'],'max_mov','masks');
    load(fullfile(dir_SNR_max,['SNR_max_',data_name,'.mat']),'SNR_max');
    load(fullfile(dir_raw_max,['raw_max_',data_name,'.mat']),'raw_max');
    [Lx,Ly] = size(raw_max);
    if use_SNR
        max_mov = SNR_max(1:Lx,1:Ly);
    else
        max_mov = raw_max;
    end
    max_mov(end-6:end-3,end-24:end-5) = inf;
%     load(fullfile(dir_Masks,['FinalMasks_',data_name,'.mat']),'FinalMasks');
    load(fullfile(dir_parent,['GT Masks updated\FinalMasks_',data_name,'_merge.mat']),'FinalMasks');

    %% show the outputs
    figure('Position',[100,50,750,500],'Color','w');
    imagesc(max_mov,color_range(ind,:)); %
    colormap gray
    hold on;
%     contour(sum(FinalMasks,3),'Color',blue);
    for n = 1:size(FinalMasks,3)
        temp = FinalMasks(:,:,n);
        if any(temp,'all')
            contour(temp, 'EdgeColor',blue,'LineWidth',0.5);
        end
    end
    axis image off;
    h=colorbar;
    if use_SNR
        set(get(h,'Label'),'String','Peak SNR','FontSize',fontsize,'FontName','Arial');
    else
        set(get(h,'Label'),'String','Maximum Intensity','FontSize',fontsize,'FontName','Arial');
    end
    title(list_titles{ind},'FontSize',fontsize);
    set(gca,'FontSize',fontsize)

    %%
    Masks_sum = zeros(size(max_mov));
    for xpart = 1:2
        for ypart = 1:2
            xrange = xyrange(ind,2*xpart-1):xyrange(ind,2*xpart);
            yrange = xyrange(ind,2*ypart-1+4):xyrange(ind,2*ypart+4);
%             FinalMasks = masks(xrange,yrange,:);
%             areas_cut = squeeze(sum(sum(FinalMasks,1),2));
%             areas_ratio = areas_cut./areas;
%             FinalMasks(:,:,areas_ratio<1/3)=[];
            mask_name = sprintf('%s\\%s\\GT Masks\\FinalMasks_%s_part%d%d.mat',dir_parent,data_name,data_name,xpart,ypart);
            Masks_part = load(mask_name,'FinalMasks');
            Masks = Masks_part.FinalMasks;
%             Masks_sum(xrange,yrange) = sum(Masks_part.FinalMasks,3);

            for n = 1:size(Masks,3)
                temp = zeros(Lx,Ly,'logical');
                temp(xrange,yrange) = Masks(:,:,n);
                if any(temp,'all')
                    contour(temp, 'EdgeColor',yellow,'LineWidth',0.5);
                end
            end
        end
    end
%     contour(Masks_sum,'Color',yellow);
    
    %%
%     plot([217,217],[1,458],'y')
%     plot([242,242],[1,458],'y')
%     plot([1,477],[153,153],'y')
%     plot([1,477],[168,168],'y')
%     plot([1,477],[321,321],'r')
    
    rectangle('Position',[xyrange(ind,5)-1,xyrange(ind,1)-1,Dimens(ind,2)+1,Dimens(ind,1)+1],'EdgeColor',green,'LineWidth',2);
    rectangle('Position',[xyrange(ind,7)-1,xyrange(ind,1)-1,Dimens(ind,2)+1,Dimens(ind,1)+1],'EdgeColor',green,'LineWidth',2);
    rectangle('Position',[xyrange(ind,5)-1,xyrange(ind,3)-1,Dimens(ind,2)+1,Dimens(ind,1)+1],'EdgeColor',green,'LineWidth',2);
    rectangle('Position',[xyrange(ind,7)-1,xyrange(ind,3)-1,Dimens(ind,2)+1,Dimens(ind,1)+1],'EdgeColor',green,'LineWidth',2);
    
    if use_SNR
        saveas(gcf,fullfile(dir_save,['Video division',data_name,' SNR.png']));
    else
        saveas(gcf,fullfile(dir_save,['Video division',data_name,' raw.png']));
    end
end