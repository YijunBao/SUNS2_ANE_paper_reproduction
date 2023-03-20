function SNRvideo = AddROI(video,name,masks,video_SNR)

global CallBackInterrupted; 
CallBackInterrupted = 0;
global IsPlaying
IsPlaying = false;   % Play flag, playing when it is 'True'

global S;
S = 1;
global sno;
global Rmin; global Rmax;
global Img;
global title_text;
global alpha;
alpha = 0.5;

Tinterv = 100;
SFntSz = 9;
txtFntSz = 10;
LVFntSz = 9;

% Images = uint16(binVideo_temporal(Images,3));   %bin to 2Hz for faster visualization
% video = (binVideo_temporal(video,3));   %bin to 2Hz for faster visualization % YB 2019/07/23
video = homo_filt(video,30);
minImages = min(video,[],3);
video = video - minImages;

meanImages = mean(video,3);
meanImages = normalizeValues(meanImages);
meanImages = imadjust(meanImages,stretchlim(meanImages,[0.25,0.99]),[],0.5); % 

corrImages = correlation_img(video);
corrImages(corrImages<0) = 0;
corrImages = imadjust(corrImages,[],[],2); % 

maxImages = max(video,[],3);
maxImages = normalizeValues(maxImages);
maxImages = imadjust(maxImages,stretchlim(maxImages,[0.25,0.99]),[],0.5); % ,[0,0.5],[]

if exist('video_SNR','var')
    SNRvideo = video_SNR;
else
    medianImages = median(video,3);
    stdImages = std(video,1,3);
    SNRvideo = (video - medianImages)./stdImages;
end
PSNRImages = max(SNRvideo,[],3);
PSNRImages = normalizeValues(PSNRImages);
PSNRImages = imadjust(PSNRImages,stretchlim(PSNRImages,[0.25,0.99]),[],0.5); % ,[0,0.5],[]

% Adjust contrast
size_video = size(video);
video = normalizeValues(video);
video1 = reshape(video,[],1);
video = reshape(imadjust(video1,stretchlim(video1,[0.1,0.99]),[],0.5),size_video);

SNRvideo = normalizeValues(SNRvideo);
size_video = size(SNRvideo);
SNRvideo1 = reshape(SNRvideo,[],1);
SNRvideo = reshape(imadjust(SNRvideo1,stretchlim(SNRvideo1,[0.25,0.995]),[],0.5),size_video);

% setup
screensize = get(groot,'ScreenSize');
fig = figure('units','normalized','outerposition',[0 0 1 1],'Visible','Off');
% set(gcf,'Position',[300,300,960,480]);
% set(gcf,'PaperPosition',[300,300,960,480]);
myhandles = guihandles(fig);
myhandles.deleteIDs = [];
guidata(fig,myhandles) 

% Add push button to Choose which image to display
pos = [1750/1920*screensize(3),700/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Mean image',...
       'Callback', @MEANIMG);
   
% Add push button to Choose which image to display
pos = [1750/1920*screensize(3),650/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Max image',...
       'Callback', @MAXIMG);
   
% Add push button to Choose which image to display
pos = [1750/1920*screensize(3),600/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Peak SNR image',...
       'Callback', @PSNRIMG);
   
  % Add push button to Choose which image to display
pos = [1750/1920*screensize(3),550/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Correlation image',...
       'Callback', @CORRIMG); 
   
   
% Add push button to give opportunity to add neurons
pos = [1750/1920*screensize(3),450/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Add neurons',...
       'Callback', @addID_right);
   
% Add push button to give opportunity to add neurons
pos = [100/1920*screensize(3),450/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Add neurons',...
       'Callback', @addID_left);
   
% Add push button to cancel the recently added neuron
pos = [940/1920*screensize(3),450/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Cancel added',...
       'Callback', @Cancel_added);
   
   
% Add push button to indicate end of work
pos = [940/1920*screensize(3),300/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Finished',...
       'Callback', @savedata);
   
% Add push button to play the raw video
pos = [100/1920*screensize(3),700/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Play raw video',...
       'Callback', @PlayRawVideo);
   
% Add push button to pause the video
pos = [100/1920*screensize(3),650/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Play/Pause',...
       'Callback', @Play);
   
% Add push button to play the SNR video
pos = [100/1920*screensize(3),600/1080*screensize(4),...
    100/1920*screensize(3),40/1080*screensize(4)];
del = uicontrol('Style','pushbutton',...
      'Position', pos,...
       'String', 'Play SNR video',...
       'Callback', @PlaySNRVideo);
   
  %%%%%%%%  
FinalMasks = masks;
[Lx,Ly,ncells] = size(FinalMasks);
greenImg = cat(3,zeros(Lx,Ly),ones(Lx,Ly),zeros(Lx,Ly));
edge_masks = cell2mat(cellfun(@edge, mat2cell(FinalMasks,Lx,Ly,ones(1,ncells)),'UniformOutput',false));
tempM = sum(edge_masks,3);
% tempM = max(FinalMasks,[],3);

% Make figure visible after adding all components
fig.Visible = 'on';
ax(2) = subplot(1,2,2);
PSNRIMG;

ax(1) = subplot(1,2,1);
Img = SNRvideo;
imagesc(squeeze(Img(:,:,S,:)));
axis('image');
title_text = 'SNR Video';
title(title_text,'FontSize',14)

sno = size(Img,3);  % number of slices
MinV = 0;
MaxV = max(Img(:));
LevV = (double( MaxV) + double(MinV)) / 2;
Win = double(MaxV) - double(MinV);
[Rmin, Rmax] = WL2R(Win, LevV);
caxis([Rmin Rmax])

if ~isempty(tempM)
    hold on;
    image(greenImg,'Alphadata',alpha*tempM);  
    hold off;
else
    tempM = zeros(Lx,Ly);
end

set(ax,'Units','pixels');
FigPos = round(get(ax(1),'Position'));
S_Pos = [FigPos(1), FigPos(2)-40, FigPos(3)+1, 20];
Stxt_Pos = [FigPos(1), FigPos(2)-20, FigPos(3)+1, 15];
Play_Pos = [FigPos(1)+FigPos(3)+40, FigPos(2)-40, 30, 20];
Time_Pos = [FigPos(1)+FigPos(3)+35, FigPos(2)-65, 40, 20];
Ttxt_Pos = [FigPos(1)+FigPos(3)-50, FigPos(2)-67, 90, 20];

% Play Button styles:
Play_BG = ones(Play_Pos(4),Play_Pos(3),3)*0.85;
Play_BG(1,:,:) = 1; Play_BG(:,1,:) = 1; Play_BG(:,end-1,:) = 0.6; Play_BG(:,end,:) = 0.4; Play_BG(end,:,:) = 0.4;
Play_Symb = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; 
             0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1; 
             0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1;
             0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1; 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1; 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1;
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1; 
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1;
             0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1; 
             0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1; 
             0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
             0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
Play_BG(floor((Play_Pos(4)-13)/2)+1:floor((Play_Pos(4)-13)/2)+13,floor(Play_Pos(3)/2)-7:floor(Play_Pos(3)/2)+6,:) = ...
    repmat(Play_Symb,[1,1,3]) .* Play_BG(floor((Play_Pos(4)-13)/2)+1:floor((Play_Pos(4)-13)/2)+13,floor(Play_Pos(3)/2)-7:floor(Play_Pos(3)/2)+6,:);
Pause_BG = ones(Play_Pos(4),Play_Pos(3),3)*0.85;
Pause_BG(1,:,:) = 1; Pause_BG(:,1,:) = 1; Pause_BG(:,end-1,:) = 0.6; Pause_BG(:,end,:) = 0.4; Pause_BG(end,:,:) = 0.4;
Pause_Symb = repmat([0, 0, 0, 1, 1, 1, 1, 0, 0, 0],13,1);
Pause_BG(floor((Play_Pos(4)-13)/2)+1:floor((Play_Pos(4)-13)/2)+13,floor(Play_Pos(3)/2)-5:floor(Play_Pos(3)/2)+4,:) = ...
    repmat(Pause_Symb,[1,1,3]) .* Pause_BG(floor((Play_Pos(4)-13)/2)+1:floor((Play_Pos(4)-13)/2)+13,floor(Play_Pos(3)/2)-5:floor(Play_Pos(3)/2)+4,:);

shand = uicontrol('Style', 'slider','Min',1,'Max',sno,'Value',S,'SliderStep',[1/(sno-1) 10/(sno-1)],'Position', S_Pos,'Callback', {@SliceSlider, Img});
stxthand = uicontrol('Style', 'text','Position', Stxt_Pos,'String',sprintf('Slice# %d / %d',S, sno), 'FontSize', SFntSz);
playhand = uicontrol('Style', 'pushbutton','Position', Play_Pos, 'Callback' , @Play);
set(playhand, 'cdata', Play_BG)
ttxthand = uicontrol('Style', 'text','Position', Ttxt_Pos,'String','Interval (ms): ',  'FontSize', txtFntSz);
timehand = uicontrol('Style', 'edit','Position', Time_Pos,'String',sprintf('%d',Tinterv), 'Background', [1 1 1], 'FontSize', LVFntSz,'Callback', @TimeChanged);

set (gcf, 'WindowScrollWheelFcn', @mouseScroll);
set(playhand, 'cdata', Play_BG)
set(shand,'Value',S);
set(stxthand, 'String', sprintf('Slice# %d / %d',S, sno));


    function MAXIMG(source,callbackdata)
        if IsPlaying
          CallBackInterrupted = 1;
        end

        subplot(ax(2));
        imshow((maxImages),[]); 
        axis image;
        colormap gray

        hold on
        if ~isempty(tempM)
            image(greenImg,'Alphadata',alpha*tempM);   
        end
        hold off
        title('Max Image','FontSize',14)
    end


    function MEANIMG(source,callbackdata)
        if IsPlaying
          CallBackInterrupted = 1;
        end

        subplot(ax(2));
        imshow((meanImages),[]); 
        axis image;
        colormap gray

        hold on
        if ~isempty(tempM)
            image(greenImg,'Alphadata',alpha*tempM);   
        end
        hold off
        title('Mean Image','FontSize',14)
    end


    function PSNRIMG(source,callbackdata)
        if IsPlaying
          CallBackInterrupted = 1;
        end

        subplot(ax(2));
        imshow((PSNRImages),[]); 
        axis image;
        colormap gray

        hold on
        if ~isempty(tempM)
            image(greenImg,'Alphadata',alpha*tempM);  
        end
        hold off
        title('Peak SNR Image','FontSize',14)
    end


    function CORRIMG(source,callbackdata)
        if IsPlaying
          CallBackInterrupted = 1;
        end

        subplot(ax(2));
        imshow((corrImages),[]); 
        axis image;
        colormap gray

        hold on
        if ~isempty(tempM)
            image(greenImg,'Alphadata',alpha*tempM);   
        end
        hold off
        title('Correlation Image','FontSize',14)
    end


    function addID_right(source,callbackdata)
        if IsPlaying
            CallBackInterrupted = 0; IsPlaying = 0; 
            set(playhand, 'cdata', Play_BG);
        end

        subplot(ax(2));
%         hFH = imfreehand();
%         if ~isempty(hFH)
        hFH = drawfreehand();
        if ~isempty(hFH) && isvalid(hFH)
            imH = imhandles(fig);
            temp = hFH.createMask(imH(1));
            if nnz(temp)
                CC = bwconncomp(temp);
                if CC.NumObjects > 1
                    areas = cellfun(@numel,CC.PixelIdxList);
                    [~, ind] = max(areas);
                    temp = 0*temp;
                    temp(CC.PixelIdxList{ind}) = 1;
                end
                
                edge_temp = edge(temp);
                FinalMasks = cat(3,FinalMasks,temp);
                edge_masks = cat(3,edge_masks,edge_temp);
                tempM = tempM + edge_temp;
    %             tempM = max(FinalMasks,[],3);
                hold on
                image(greenImg,'Alphadata',alpha*edge_temp);  
            end
        end
    end


    function addID_left(source,callbackdata)
        if IsPlaying
            CallBackInterrupted = 0; IsPlaying = 0; 
            set(playhand, 'cdata', Play_BG);
        end

        subplot(ax(1));
        hFH = drawfreehand();
        if ~isempty(hFH) && isvalid(hFH)
            imH = imhandles(fig);
            temp = hFH.createMask(imH(1));
            if nnz(temp)
                CC = bwconncomp(temp);
                if CC.NumObjects > 1
                    areas = cellfun(@numel,CC.PixelIdxList);
                    [~, ind] = max(areas);
                    temp = 0*temp;
                    temp(CC.PixelIdxList{ind}) = 1;
                end
                
                edge_temp = edge(temp);
                FinalMasks = cat(3,FinalMasks,temp);
                edge_masks = cat(3,edge_masks,edge_temp);
                tempM = tempM + edge_temp;
    %             tempM = max(FinalMasks,[],3);
                hold on
                image(greenImg,'Alphadata',alpha*edge(temp));  
            end
        end
    end


    function Cancel_added(source,callbackdata)
        if IsPlaying
%             CallBackInterrupted = 1;
            CallBackInterrupted = 1; IsPlaying = 0; 
            set(playhand, 'cdata', Play_BG);
        end

        if ~isempty(FinalMasks)
            FinalMasks(:,:,end) = [];
            edge_temp = edge_masks(:,:,end);
            tempM = tempM - edge_temp;
            edge_masks(:,:,end) = [];
        end
        
%         tempM = max(FinalMasks,[],3);
        hold off
        imagesc(squeeze(Img(:,:,S,:)));
        axis('image');
        caxis([Rmin Rmax])
        title(title_text,'FontSize',14)
        hold on
        image(greenImg,'Alphadata',alpha*tempM);  
    end


    function PlayRawVideo(source,callbackdata)
        Img = video;
        title_text = 'Minimum Subtracted Video';
        sno = size(Img,3);  % number of slices
        MinV = 0;
        MaxV = max(Img(:));
        LevV = (double( MaxV) + double(MinV)) / 2;
        Win = double(MaxV) - double(MinV);
        [Rmin, Rmax] = WL2R(Win, LevV);

        subplot(ax(1));
        cla
        imagesc(squeeze(Img(:,:,S,:)));
        axis('image');
        caxis([Rmin Rmax])
        title(title_text,'FontSize',14)

        if ~isempty(tempM)
            hold on;
            image(greenImg,'Alphadata',alpha*tempM); 
            hold off;
        end

%         IsPlaying = 0;
%         Play;
%         if CallBackInterrupted 
%             CallBackInterrupted = 0; IsPlaying = 0; 
%             set(playhand, 'cdata', Play_BG);
%             return;
%         end        
    end


    function PlaySNRVideo(source,callbackdata)
        Img = SNRvideo;
        title_text = 'SNR Video';
        sno = size(Img,3);  % number of slices
        MinV = 0;
        MaxV = max(Img(:));
        LevV = (double( MaxV) + double(MinV)) / 2;
        Win = double(MaxV) - double(MinV);
        [Rmin, Rmax] = WL2R(Win, LevV);
        
        subplot(ax(1));
        cla
        imagesc(squeeze(Img(:,:,S,:)));
        axis('image');
        caxis([Rmin Rmax])
        title(title_text,'FontSize',14)

        if ~isempty(tempM)
            hold on;
            image(greenImg,'Alphadata',alpha*tempM); 
            hold off;
        end

%         IsPlaying = 0;
%         Play;
%         if CallBackInterrupted 
%             CallBackInterrupted = 0; IsPlaying = 0; 
%             set(playhand, 'cdata', Play_BG);
%         end        
    end


    function savedata(source,callbackdata)
        if IsPlaying
          CallBackInterrupted = 1;
        end

        FinalMasks = logical(FinalMasks);
        save(name,'FinalMasks');
        close(fig)
    end


% -=< Slice slider callback function >=-
    function SliceSlider (hObj,event, ~)
        subplot(ax(1));
        S = round(get(hObj,'Value'));
        imagesc(squeeze(Img(:,:,S,:)));
        axis('image');
        caxis([Rmin Rmax])
        title(title_text,'FontSize',14)

        if ~isempty(tempM)
            hold on;
            image(greenImg,'Alphadata',alpha*tempM); 
            hold off;
        end
        if sno > 1
            set(stxthand, 'String', sprintf('Slice# %d / %d',S, sno));
        else
            set(stxthand, 'String', '2D image');
        end
    end


% -=< Mouse scroll wheel callback function >=-
    function mouseScroll (object,eventdata)
        subplot(ax(1));
        UPDN = eventdata.VerticalScrollCount;
        S = S - UPDN;
        if (S < 1)
            S = 1;
        elseif (S > sno)
            S = sno;
        end
        if sno > 1
            set(shand,'Value',S);
            set(stxthand, 'String', sprintf('Slice# %d / %d',S, sno));
        else
            set(stxthand, 'String', '2D image');
        end
        imagesc(squeeze(Img(:,:,S,:)));
        axis('image');
        caxis([Rmin Rmax])
        title(title_text,'FontSize',14)

        if ~isempty(tempM)
            hold on;
            image(greenImg,'Alphadata',alpha*tempM); 
            hold off;
        end
    end


% -=< Play button callback function >=-
    function Play (hObj,event)
        subplot(ax(1));
        IsPlaying = ~IsPlaying;
        if IsPlaying
            set(playhand, 'cdata', Pause_BG)
        else
            set(playhand, 'cdata', Play_BG)
        end            

        while IsPlaying
            S = S + 1;
            if (S > sno)
                S = 1;
            end
            set(shand,'Value',S);
            set(stxthand, 'String', sprintf('Slice# %d / %d',S, sno));
            imagesc(squeeze(Img(:,:,S,:)));
            axis('image');
            caxis([Rmin Rmax])
            title(title_text,'FontSize',14)

            if ~isempty(tempM)
                hold on;
                image(greenImg,'Alphadata',alpha*tempM); 
                hold off;
            end

            pause(Tinterv/1000)
            if CallBackInterrupted 
                CallBackInterrupted = 0; IsPlaying = 0; 
                set(playhand, 'cdata', Play_BG);
                return;
            end        
        end
    end


% -=< Window and level to range conversion >=-
    function [Rmn, Rmx] = WL2R(W,L)
        Rmn = L - (W/2);
        Rmx = L + (W/2);
        if (Rmn >= Rmx)
            Rmx = Rmn + 1;
        end
    end


% -=< Time interval adjustment callback function>=-
    function TimeChanged(varargin)
        Tinterv = str2double(get(timehand, 'string'));
    end
    
end