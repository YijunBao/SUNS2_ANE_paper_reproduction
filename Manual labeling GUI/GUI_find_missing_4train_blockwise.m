function GUI_find_missing_4train_blockwise(video, folder, masks, patch_locations, ...
    images_added_crop, masks_added_crop, added_frames, added_weights, update_result)
% constants
global fileName;
global thred_IoU;
global thred_IoU_split;
global thred_jaccard_inclass;
global thred_jaccard;
% global auto_on;
% global r_bg_ext;
% r_bg_ext = 24;
% Leng = 2*r_bg_ext+1;

% global variables calculated outside the GUI
% global list_weight;
% global list_weight_trace;
% global list_weight_frame;
% global traces_raw;
% global traces_out_exclude;
% global traces_bg_exclude;
global area;
% global r_bg_ratio;
% global comx;
% global comy;
global masks_neighbors; 
global avg_radius;

% global variables for each neuron
global xmin;
global ymin;
global xmax;
global ymax;
global mask_sub;
global mask_update;
global video_sub;
global xrange;
global yrange;
global Lxm;
global Lym;
global weight;
global select_frames_sort;
global select_frames_sort_full;
global select_frames_order;
global n_select;
global n_class;
global avg_frame; 
global confirmed_frames;
global select_frames_class;
global select_weight_calss;
global classes;
global nearby_disk;
global unmasked;

% global variables related to display
global gui;
global video_sub_sort;
global mask_update_tile;
global mask_update_select;
global edges;
global sum_edges;
global active_ax;
global list_markers;
global extra_button;
global list_valid;
global current_thred;
global new_contour;

% global variables summarizing results
global masks_update; 
global list_processed;
global list_added;
global ListStrings;
% global list_IoU;
global list_avg_frame;
global list_mask_update; 
global list_class_frames;
global list_noisy;
global list_far;
global masks_sum;
global list_added_frames;
global list_added_weights;
global list_select_frames;

% global select_frames;
% global mask_update_select;
% global ncol; 
% global num_avg; 
% global nrow;

[Lx,Ly,T] = size(video);
[Lxm, Lym, N] = size(masks_added_crop);
% [Lx,Ly,N] = size(masks);
ncol = 10;
num_avg = max(60, min(90, ceil(T*0.01)));
nrow = ceil(num_avg/ncol);
tile_size = [nrow, ncol];

thred_IoU_split = 0.5;
thred_jaccard_inclass = 0.4;
thred_jaccard = 0.7;
area = squeeze(sum(sum(masks,1),2));
avg_area = median(area);
avg_radius = sqrt(avg_area/pi);
masks_sum = sum(masks,3);

if exist('update_result','var')
    list_processed = update_result.list_processed;
    ListStrings = update_result.ListStrings;
    list_valid = update_result.list_valid;
    list_select_frames = update_result.list_select_frames;
    list_mask_update = update_result.list_mask_update;
else
    list_processed = false(1,N);
    ListStrings = num2cell(1:N);
    [list_valid, list_select_frames, list_mask_update] = deal(cell(1,N));
end

%%
global color;
color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% color = color([2,3,4,6,7],:);
txtFntSz = 10;

maxImages = max(video,[],3);
q = 1-avg_area/(Lxm*Lym);

nn = find(~list_processed,1); %n = current neuron
if isempty(nn)
    nn = N;
end

%% Text messages
text_left1 = {'Red: Intensity isoheight in this frame.',...
    'Selected frames are sorted according to their weight.',...
    'The frame index "m_" and "_n" means this frame has the mn-th weight (starting from 00).'};
text_OX = {'O: Approved frame used for averaging; ',...
    'X: Disgarded frame not used for averaging. '};
text_center = {'Red: New mask',...
    'Other colors: Original neurons'};
text_right_start = {'Do you want to update the mask of this neuron? ',...
    'Click "Save results" to save the current progress and exit the GUI.'};
text_right_continue = {['Click the frames that you want to change assignment. ',...
    'Each clicked frame will change assignment in cyclical. ',...
    'Press "Esc" to finish selection.']};
text_right_confirm = {'Click "Select Frames" to edit the selected frames for averaging.',...
    'Click "True" to confirm that the red contour on the "Average Image" panel is a true missing neuron. ',...
    'Click "False" to record that this is not a missing neuron. ',...
    'Click "Smaller area" or "Larger area" to adjust the contour threshold.'};

%% -------------------------------------------------------------------------%
gui = createInterface();
updateInterface();

%%
function gui = createInterface()
    txtFntSz = 11;
    gui = struct();
    screensize = get(groot, 'ScreenSize');
    screensize = screensize + [0,50,-50,-100];
    gui.Window = figure(...
        'Name', 'Neuron Contour Refinement', ...
        'NumberTitle', 'off', ...
        'MenuBar', 'none', ...
        'Toolbar', 'none', ...
        'HandleVisibility', 'off', ...
        'Position', screensize ...
        );

    % Arrange the main interface
    gui.mainLayout = uix.HBoxFlex(...
    'Parent', gui.Window, ...
        'Spacing', 3);
    gui.LeftLayout = uix.VBoxFlex(...
        'Parent', gui.mainLayout, ...
        'Padding', 3);
    gui.CenterLayout = uix.VBoxFlex(...
        'Parent', gui.mainLayout, ...
        'Padding', 3);
    gui.RightLayout = uix.VBoxFlex(...
        'Parent', gui.mainLayout, ...
        'Padding', 3);
    set(gui.mainLayout, 'Width', [- 10, - 4, - 2]);

    % Left Layout Design
    gui.FramesPanel = uipanel(...
    'Parent', gui.LeftLayout, ...
        'FontSize', txtFntSz, ...
        'Title', 'Selected Frames');
    gui.FramesAxes = axes('Parent', gui.FramesPanel);
    Frames_position = gui.FramesPanel.Position;
%     button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
%     gui.Confirm = uicontrol(...
%         'Parent', gui.FramesPanel, ...
%         'Position', button1_position, ...
%         'Style', 'PushButton', ...
%         'String', 'Confirm Selection', ...
%         'FontSize', txtFntSz);
%     button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
%     gui.Change = uicontrol(...
%         'Parent', gui.FramesPanel, ...
%         'Position', button2_position, ...
%         'Style', 'PushButton', ...
%         'String', 'Change Selection', ...
%         'FontSize', txtFntSz);
%     button3_position = [Frames_position(1)+Frames_position(3)-600, 10, 190, 50];
%     gui.Single = uicontrol(...
%         'Parent', gui.FramesPanel, ...
%         'Position', button3_position, ...
%         'Style', 'PushButton', ...
%         'String', 'Single Neuron', ...
%         'FontSize', txtFntSz, ...
%         'CallBack', @Single);
    ContoursButton_position = [Frames_position(1), 10, 250, 50];
    gui.ShowContoursButton = uicontrol(...
        'Style', 'ToggleButton', ...
        'Parent', gui.FramesPanel, ...
        'Position', ContoursButton_position, ...
        'String', 'Hide contours in each frame', ...
        'FontSize', txtFntSz, ...
        'Value', 1, ...
        'CallBack', @ShowContours);
%     PlayButton_position = [Frames_position(1)+260, 10, 150, 50];
%     gui.PlayTransientButton = uicontrol(...
%         'Style', 'ToggleButton', ...
%         'Parent', gui.FramesPanel, ...
%         'Position', PlayButton_position, ...
%         'String', 'Paly a transient', ...
%         'FontSize', txtFntSz, ...
%         'Value', 1, ...
%         'CallBack', @PlayTransient);

    gui.InfoBox = uix.HBoxFlex(...
        'Parent', gui.LeftLayout, ...
        'Padding', 3);
    gui.InfoTextLeft1 = uicontrol(...
        'Style', 'text',...
        'Parent', gui.InfoBox, ...
        'HorizontalAlignment', 'left', ...
        'FontSize', txtFntSz-1, ...
        'String', text_left1);
    gui.InfoTextLeft2 = uicontrol(...
        'Style', 'text',...
        'Parent', gui.InfoBox, ...
        'HorizontalAlignment', 'left', ...
        'FontSize', txtFntSz-1, ...
        'String', text_OX);
    set(gui.LeftLayout, 'Heights', [- 10, - 1]);

    % Center Layout Design
    gui.SummaryPanel = uipanel(...
        'Parent', gui.CenterLayout, ...
        'FontSize', txtFntSz, ...
        'Title', 'Peak SNR Over Full Video');
    gui.SummaryAxes = axes(...
        'Parent', gui.SummaryPanel);

    gui.AvgPanel = uipanel(...
        'Parent', gui.CenterLayout, ...
        'FontSize', txtFntSz, ...
        'Title', 'Averaged Image');
    gui.AvgAxes = axes(...
        'Parent', gui.AvgPanel);

    gui.ThresholdPanel = uix.HBoxFlex(...
        'Parent', gui.CenterLayout, ...
        'Padding', 3);
    gui.HigherThreshold = uicontrol(...
        'Parent', gui.ThresholdPanel, ...
        'Style', 'PushButton', ...
        'String', 'Smaller area', ...
        'FontSize', txtFntSz, ...
        'CallBack', @HigherThreshold);
    gui.LowerThreshold = uicontrol(...
        'Parent', gui.ThresholdPanel, ...
        'Style', 'PushButton', ...
        'String', 'Larger area', ...
        'FontSize', txtFntSz, ...
        'CallBack', @LowerThreshold);

    gui.ClimPanel = uix.HBoxFlex(...
        'Parent', gui.CenterLayout, ...
        'Padding', 3);
    gui.InfoTextCenter = uicontrol(...
        'Style', 'text',...
        'Parent', gui.ClimPanel, ...
        'HorizontalAlignment', 'left', ...
        'FontSize', txtFntSz-1, ...
        'String', text_center);
    gui.HigherClim = uicontrol(...
        'Parent', gui.ClimPanel, ...
        'Style', 'PushButton', ...
        'String', 'Larger colorbar range', ...
        'FontSize', txtFntSz, ...
        'CallBack', @HigherClim);
    set(gui.CenterLayout, 'Heights', [- 4.5, - 5, - .5, - 1]);

%     gui.AvgTabGroup = uitabgroup(...
%         'Parent', gui.AvgPanel);
% %     delete(gui.Confirm);
% %     delete(gui.Change)
% % %     delete(gui.Single)
% %     delete(gui.AvgTabGroup)
%     gui.masks_update_tile = gui.AvgTabGroup;
%     delete(gui.masks_update_tile);

    % Right Layout Design
    gui.NeuronPanel = uix.VBoxFlex(...
        'Parent', gui.RightLayout);
    gui.ListPanel = uix.BoxPanel(...
        'Parent', gui.NeuronPanel, ...
        'FontSize', txtFntSz, ...
        'Title', 'Neuron List');
%     ListStrings = num2cell(1:N);
    gui.ListBox = uicontrol(...
        'Style', 'ListBox', ...
        'Parent', gui.ListPanel, ...
        'FontSize', 10, ...
        'String', ListStrings, ...
        'Value', nn, ...
        'CallBack', @MoveToNeuron);
    gui.ListFB = uix.HBoxFlex(...
        'Parent', gui.NeuronPanel, ...
        'Padding', 3);
    gui.ListForward = uicontrol(...
        'Parent', gui.ListFB, ...
        'Style', 'PushButton', ...
        'String', 'Forward Neuron', ...
        'CallBack', @Forward);
    gui.ListBackward = uicontrol(...
        'Parent', gui.ListFB, ...
        'Style', 'PushButton', ...
        'String', 'Backward Neuron', ...
        'CallBack', @Backward);
    set(gui.NeuronPanel, 'Heights', [- 5, - 1]);

    gui.ControlPanelTitle = uix.BoxPanel(...
        'Parent', gui.RightLayout, ...
        'Padding', 3, ...
        'FontSize', txtFntSz, ...
        'Title', 'Select Action');
    gui.ControlPanel = uix.VBoxFlex(...
        'Parent', gui.ControlPanelTitle);
    gui.SaveMasks = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'Save results', ...
        'FontSize', txtFntSz, ...
        'CallBack', @SaveMasks);

    gui.InfoPanel = uix.BoxPanel(...
        'Parent', gui.RightLayout, ...
        'Padding', 3, ...
        'FontSize', txtFntSz, ...
        'Title', 'Action Instruction');
    gui.InfoText = uicontrol(...
        'Style', 'edit',...
        'Parent', gui.InfoPanel, ...
        'HorizontalAlignment', 'left', ...
        'FontSize', txtFntSz-1, ...
        'String', 'Info',...
        'Max',14);

    gui.IoUPanel = uix.BoxPanel(...
        'Parent', gui.RightLayout, ...
        'Padding', 3, ...
        'FontSize', txtFntSz, ...
        'Title', 'Automatically approve when');
    gui.IoULabel = uix.HBoxFlex(...
        'Parent', gui.IoUPanel, ...
        'Padding', 3);
%     gui.IoUgt = uicontrol(...
%         'Parent', gui.IoULabel, ...
%         'Style', 'text', ...
%         'String', 'IoU >', ...
%         'FontSize', txtFntSz, ...
%         'HorizontalAlignment', 'right');
%     gui.IoUInput = uicontrol(...
%         'Parent', gui.IoULabel, ...
%         'Style', 'edit', ...
%         'String', '0.7', ...
%         'FontSize', txtFntSz, ...
%         'InnerPosition', [5,5,100,15], ...
%         'HorizontalAlignment', 'left');
%     gui.ChangeIoUth = uicontrol(...
%         'Style', 'PushButton', ...
%         'Parent', gui.IoULabel, ...
%         'String', 'Change', ...
%         'FontSize', txtFntSz, ...
%         'CallBack', @ChangeIoUth);
    gui.TrueNeuron = gui.IoULabel;
    delete(gui.TrueNeuron);
    gui.FalseNeuron = gui.IoULabel;
    delete(gui.FalseNeuron);
    gui.SelectFrames = gui.IoULabel;
    delete(gui.SelectFrames);

    set(gui.RightLayout, 'Heights', [- 4, - 1, - 4, - 1]);

    extra_button = 0;
    thred_IoU = 0.5;
end

%% updateInterface - update GUI for new trace
function updateInterface()
    delete(gui.TrueNeuron);
    delete(gui.FalseNeuron)
    delete(gui.SelectFrames)
%     delete(gui.Single)
%     delete(gui.AvgTabGroup)
%     delete(gui.FinishInfo);
    confirmed_frames = [];
    list_markers = {};
    extra_button = 0;
    
    %% Update the average image
%     mask = masks(:,:,nn);
%     r_bg = sqrt(mean(area)/pi)*r_bg_ratio;
%     r_bg_ext = r_bg*(r_bg_ratio+1)/r_bg_ratio;
    list_locations = patch_locations(nn,:);
    xmin = list_locations(1);
    xmax = list_locations(2);
    ymin = list_locations(3);
    ymax = list_locations(4);
    xrange = xmin:xmax;
    yrange = ymin:ymax;
    masks_sub = masks(xrange,yrange,:);
%     masks_sum_sub = sum(masks_sub,3);
    neighbors = squeeze(sum(sum(masks_sub,1),2)) > 0;
    masks_neighbors = masks_sub(:,:,neighbors);
    unmasked = ~sum(masks_neighbors,3);
    video_sub = video(xrange,yrange,:);
%     [Lxm, Lym, ~] = size(video_sub_sort);
    
    weight = added_weights{nn};
    avg_frame = images_added_crop(:,:,nn);
    thred_inten = quantile(avg_frame, q, 'all');
    [~, ~, current_thred] = threshold_frame(avg_frame, thred_inten, avg_area, unmasked);
    mask_update = masks_added_crop(:,:,nn);
    
    video_sub_sort = added_frames{nn};
    n_select = size(video_sub_sort,3);
    select_frames_order = 1:n_select;

    updateAverage();
    updateTiles();
%     updateTiles(max(avg_frame(mask_sub)));
    
    %% Update summary image
    axes(gui.SummaryAxes);
    cla(gui.SummaryAxes);
    imagesc(gui.SummaryAxes,maxImages(xrange,yrange),[0,10]);
    colormap(gui.SummaryAxes, 'gray');
    colorbar(gui.SummaryAxes);
    axis(gui.SummaryAxes,'image');
    hold(gui.SummaryAxes, 'on');
%     neighbors = list_neighbors{nn};
    for kk = 1:size(masks_neighbors,3)
        contour(gui.SummaryAxes,masks_neighbors(:,:,kk),'Color',color(mod(kk,size(color,1))+1,:));
    end
%     contour(gui.SummaryAxes,mask_sub,'b');
    contour(gui.SummaryAxes,mask_update,'r');
%     image_green = zeros(Lxm,Lym,3,'uint8');
%     image_green(:,:,2)=255;
%     edge_others = sum_edges(xrange,yrange)-sum(edges(xrange,yrange,[nn,neighbors]),3);
%     imagesc(gui.SummaryAxes,image_green,'alphadata',0.5*edge_others);
    title(gui.SummaryAxes,'Peak SNR (entire video)')
    
    %% Update the button group
    set(gui.InfoText, 'String', text_right_start);

    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.TrueNeuron = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'True', ...
        'FontSize', txtFntSz, ...
        'CallBack', @TrueNeuron);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.FalseNeuron = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'False', ...
        'FontSize', txtFntSz, ...
        'CallBack', @FalseNeuron);
    button3_position = [Frames_position(1)+Frames_position(3)-600, 10, 190, 50];
    gui.SelectFrames = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button3_position, ...
        'Style', 'PushButton', ...
        'String', 'Select Frames', ...
        'FontSize', txtFntSz, ...
        'CallBack', @SelectFrames);
    extra_button = 5;
    set(gui.InfoText, 'String', text_right_confirm);
    
%     MultipleNeurons();
end

%% Update selected frames
function updateTiles()
%     if IoU<0.6 % || length(select_frames)<0.01*T
%     n_select = length(select_frames_sort);
%     video_sub_sort = video_sub(:,:,select_frames_sort);
    images_tile = imtile(video_sub_sort, 'GridSize', tile_size);
%     mask_sub_tile = repmat(mask_sub,tile_size);
    mask_update_select = false(Lxm,Lym,n_select);
    list_noisy = false(1,n_select);
    for kk = 1:n_select
        frame = video_sub_sort(:,:,kk);
%         frame(~nearby_disk) = 0;
        thred_inten = quantile(frame, q, 'all');
        [frame_thred, noisy] = threshold_frame(frame, thred_inten, avg_area, unmasked);
        mask_update_select(:,:,kk) = frame_thred;
        list_noisy(kk) = noisy;
    end
    
%         video_sub_2 = reshape(video_sub(:,:,select_frames(select_frames_order)),Lxm*Lym,n_select);
%         mask_sub_2 = reshape(mask_sub,Lxm*Lym,1);
%         max_inten_each = max(video_sub_2(mask_sub_2,:),[],1);
%         mask_update_select = reshape(video_sub_2>max_inten_each*thred_binary,Lxm,Lym,n_select);
    mask_update_tile = imtile(mask_update_select, 'GridSize', tile_size);
%         mask_update_tile = repmat(mask_update,tile_size);
%         edge_others = sum_edges(xrange,yrange)-edge(mask_sub);
%         edge_others_tile = repmat(edge_others,tile_size);

%     max_inten = max(avg_frame(mask_sub));
    axes(gui.FramesAxes);
    cla(gui.FramesAxes);
%     if ~exist('max_inten','var')
%         max_inten = prctile(video_sub_sort, 99,'all');
%     end
%     if n_select > 0
%         imagesc(gui.FramesAxes, images_tile,[-1,max_inten]);
%     else
%         imagesc(gui.FramesAxes, images_tile,[-1,1]);
%     end
    imagesc(gui.FramesAxes, images_tile,[-1,3]);
    colormap(gui.FramesAxes,'gray')
    colorbar(gui.FramesAxes);
    axis(gui.FramesAxes,'image');
    hold(gui.FramesAxes,'on');
%         image_green = zeros(size(images_tile,1),size(images_tile,2),3,'uint8');
%         image_green(:,:,2)=255;
%         imagesc(image_green,'alphadata',0.5*edge_others_tile);
%     contour(gui.FramesAxes,mask_sub_tile,'b');
    if gui.ShowContoursButton.Value
        [~,gui.masks_update_tile] = contour(gui.FramesAxes,mask_update_tile,'r');
    end
%     list_frame = cell(1,n_select);
%     for kk = 1:n_select
%         [yn,xn] = ind2sub([ncol,nrow],kk);
%         xx = (xn)*Lxm-6;
%         yy = (yn-1)*Lym+4;
%         list_frame{kk} = text(gui.FramesAxes,yy,xx,num2str(select_frames_sort(kk)),'FontSize',10,'Color','y');
%     end
    title(gui.FramesAxes,['Neuron ',num2str(nn)]);
    set(gui.FramesAxes,...
        'XTick',round(Lym*(0.5:ncol-0.5)), 'YTick',round(Lxm*(0.5:nrow-0.5)), ...
        'XTickLabel',arrayfun(@(x) ['_',num2str(x)], 0:ncol-1, 'UniformOutput',false),...
        'YTickLabel',arrayfun(@(x) [num2str(x),'_'], 0:nrow-1, 'UniformOutput',false),...
        'TickLabelInterpreter','None');
%     set(gui.InfoTextLeft2, 'String', '');
end    

%%
function ShowContours(~,~)
    if gui.ShowContoursButton.Value
        set(gui.ShowContoursButton, 'String', 'Hide contours in each frame');
        [~,gui.masks_update_tile] = contour(gui.FramesAxes,mask_update_tile,'r');
    else
        set(gui.ShowContoursButton, 'String', 'Show contours in each frame');
%         if isfield(gui,'masks_update_tile')
            delete(gui.masks_update_tile);
%         end
    end
end    

%% Update average frames
function updateAverage()
    axes(gui.AvgAxes);
    cla(gui.AvgAxes);
    imagesc(gui.AvgAxes,avg_frame,[-1,3]);
    colormap(gui.AvgAxes,'gray');
    colorbar(gui.AvgAxes);
    axis(gui.AvgAxes,'image');
    hold(gui.AvgAxes,'on');
%     neighbors = list_neighbors{nn};
    for kk = 1:size(masks_neighbors,3)
        contour(gui.AvgAxes,masks_neighbors(:,:,kk),'Color',color(mod(kk,size(color,1))+1,:));
    end
    [~, new_contour] = contour(gui.AvgAxes,mask_update,'r');
    title(gui.AvgAxes,'Average frame')
%     current_thred = q;
%     tit_posi = get(tit,'Position');
end 

%% Forward - move current neuron forward 1 and update interface
function Forward(~, ~)
    nn = nn + 1;
    if nn > N
        nn = N;
    else
        set(gui.ListBox, 'Value', nn);
        updateInterface();
    end
end

%% Backward - move current neuron backward 1 and update interface
function Backward(~, ~)
    nn = nn - 1;
    if nn < 1
        nn = 1;
    else
        set(gui.ListBox, 'Value', nn);
        updateInterface();
    end
end

%% 
function MoveToNeuron(src, ~)
    persistent chk
    if isempty(chk)
        chk = 1;
        pause(0.5); %Add a delay to distinguish single click from a double click
        if chk == 1
            chk = [];
        end
    else
        chk = [];
        nn = get(src, 'Value');
        updateInterface();
        set(gui.ListBox, 'Value', nn);
%         set(gui.ListBox, 'String', ListStrings);
    end
end

%% YesSpike - Check if user is looking for spikes, print to output file,
%and play video for next spike
% function Confirm(~, ~)
%     list_mask_update{nn} = mask_update;
%     masks_update(xrange,yrange,nn) = mask_update;
%     ListStrings{nn} = [num2str(nn),' Updated'];
%     
%     set(gui.ListBox, 'String', ListStrings);
%     Forward();
% end

%% NoSpike - Check if user is looking for spikes, print to output file,
%and play video for next spike
% function Reject(~, ~)
%     list_mask_update{nn} = mask_sub;
%     masks_update(xrange,yrange,nn) = mask_sub;
%     ListStrings{nn} = [num2str(nn),' Rejected'];
%     
%     set(gui.ListBox, 'String', ListStrings);
%     Forward();
% end

%%
function ind = select_frame()
    roi = drawpoint(gui.FramesAxes);
    if isvalid(roi)
        Position = roi.Position;
        if ~isempty(Position)
            xn1 = ceil(Position(2)/Lxm);
            yn1 = ceil(Position(1)/Lym);
            ind = (xn1-1)*ncol+yn1;
        else
            ind = nan;
        end
    else
        ind = nan;
    end
%     [x1,y1] = ginput(1);
%     xn1 = ceil(x1/Lxm);
%     yn1 = ceil(y1/Lym);
%     xyn1 = (yn1-1)*ncol+xn1;
end

%%
function LowerThreshold(~,~)
    avg_frame_use = avg_frame;
    [mask_update, ~, current_thred] = threshold_frame(avg_frame_use, current_thred*0.9, [], unmasked);
    delete(new_contour);
    [~, new_contour] = contour(gui.AvgAxes,mask_update,'r');
end

%%
function HigherThreshold(~,~)
    avg_frame_use = avg_frame;
    [mask_update, ~, current_thred] = threshold_frame(avg_frame_use, current_thred*1.1, [], unmasked);
    delete(new_contour);
    [~, new_contour] = contour(gui.AvgAxes,mask_update,'r');
end

%%
function HigherClim(~,~)
    Clim = get(gui.AvgAxes,'Clim');
    Clim(2) = Clim(2) + 1;
    set(gui.AvgAxes,'Clim',Clim);
    set(gui.FramesAxes,'Clim',Clim);
end

%%
% function PlayTransient(~,~)
% %     delete(gui.Play);
%     Frames_position = gui.FramesPanel.Position;
%     esc_position = [Frames_position(1)+Frames_position(3)-400, 10, 390, 50];
%     gui.FinishInfo = uicontrol(...
%         'Parent', gui.FramesPanel, ...
%         'Position', esc_position, ...
%         'Style', 'text', ...
%         'String', 'Click the transient to play', ...
%         'FontSize', 20);
%     extra_button = 0;
% 
% %     set(gui.InfoText, 'String', text_right_manual_continue);
% %     set(gui.InfoTextLeft2, 'String', text_OX);
%     ind = select_frame();
%     kk = select_frames_sort_full(ind);
%     transient_play = video_sub(:,:,max(1,kk-20):min(T,kk+40));
%     figure(110);
%     imshow3D(transient_play,get(gui.FramesAxes,'Clim'),min(kk,21));
%     delete(gui.FinishInfo);
% end

%%
function SelectFrames(~,~)
    delete(gui.TrueNeuron);
    delete(gui.FalseNeuron)
    delete(gui.SelectFrames)
    if isempty(confirmed_frames)
        confirmed_frames = true(1,n_select);
        list_markers = cell(1,n_select);
        for ind = 1:n_select
            [yn,xn] = ind2sub([ncol,nrow],ind);
            xx = (xn-1)*Lxm+4;
            yy = (yn-1)*Lym+4;
            list_markers{ind} = plot(gui.FramesAxes,yy,xx,'yo','MarkerSize',10);
        end
    end
    Frames_position = gui.FramesPanel.Position;
    esc_position = [Frames_position(1)+Frames_position(3)-400, 10, 390, 50];
    gui.FinishInfo = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', esc_position, ...
        'Style', 'text', ...
        'String', 'Press "Esc" to finish', ...
        'FontSize', 20);
    extra_button = 0;

    set(gui.InfoText, 'String', text_right_continue);
%     set(gui.InfoTextLeft2, 'String', text_OX);
    ind = select_frame();
    while ~isnan(ind)
        if ind <= n_select
            confirmed_frames(ind) = ~confirmed_frames(ind);
            delete(list_markers{ind})
            [yn,xn] = ind2sub([ncol,nrow],ind);
            xx = (xn-1)*Lxm+4;
            yy = (yn-1)*Lym+4;
            if confirmed_frames(ind)
                list_markers{ind} = plot(gui.FramesAxes,yy,xx,'yo','MarkerSize',10);
            else
                list_markers{ind} = plot(gui.FramesAxes,yy,xx,'yx','MarkerSize',10);
            end
        end
        ind = select_frame();
    end
    delete(gui.FinishInfo);

%     select_frames_sort = {select_frames_sort_full(classes==1),...
%         select_frames_sort_full(classes==2)};
%     [avg_frame,mask_update,select_frames_class,select_weight_calss] = deal(cell(1,n_class));
%     for c = 1:n_class
%         list_class_frames{c} = (classes==c);
%         select_frames_sort{c} = select_frames_sort_full(classes==c);
%         updateAverageTab(c);
%     end
%     list_avg_frame{nn} = avg_frame;

    select_frames_sort = video_sub_sort(:,:,confirmed_frames);
    weight_select = weight(confirmed_frames);
    avg_frame = sum(select_frames_sort.*reshape(weight_select,1,1,[]),3)...
        /sum(weight_select);
%     avg_frame = mean(select_frames_sort,3);
    thred_inten = quantile(avg_frame, q, 'all');
    [mask_update, ~, current_thred] = threshold_frame(avg_frame, thred_inten, avg_area, unmasked);
    
    updateAverage();
    
    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.TrueNeuron = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'True', ...
        'FontSize', txtFntSz, ...
        'CallBack', @TrueNeuron);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.FalseNeuron = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'False', ...
        'FontSize', txtFntSz, ...
        'CallBack', @FalseNeuron);
    button3_position = [Frames_position(1)+Frames_position(3)-600, 10, 190, 50];
    gui.SelectFrames = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button3_position, ...
        'Style', 'PushButton', ...
        'String', 'Select Frames', ...
        'FontSize', txtFntSz, ...
        'CallBack', @SelectFrames);
    extra_button = 5;
    set(gui.InfoText, 'String', text_right_confirm);
end

%%
function TrueNeuron(~,~)
    valid = true;
    list_valid{nn} = valid;
    list_select_frames{nn} = find(confirmed_frames);
    list_mask_update{nn} = mask_update;
    confirmed_frames = [];
    list_markers = {};
    
    clear select_frames_sort_full;
    delete(gui.TrueNeuron);
    delete(gui.FalseNeuron);
    delete(gui.SelectFrames)
%     delete(gui.Single)
%     delete(gui.AvgTabGroup)
    extra_button = 0;
    
    list_processed(nn) = true;
    ListStrings{nn} = [num2str(nn),' True'];
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%%
function FalseNeuron(~,~)
    valid = false;
    list_valid{nn} = valid;
    list_select_frames{nn} = find(confirmed_frames);
    list_mask_update{nn} = mask_update;
    confirmed_frames = [];
    list_markers = {};
    
    clear select_frames_sort_full;
    delete(gui.TrueNeuron);
    delete(gui.FalseNeuron);
    delete(gui.SelectFrames)
%     delete(gui.Single)
%     delete(gui.AvgTabGroup)
    extra_button = 0;
    
    list_processed(nn) = true;
    ListStrings{nn} = [num2str(nn),' False'];
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%% SaveTrace - saves labeling output file and closes GUI window
function SaveMasks(~, ~)
    update_result.list_processed = list_processed;
    update_result.ListStrings = ListStrings;
    update_result.list_valid = list_valid;
    update_result.list_select_frames = list_select_frames;
    update_result.list_mask_update = list_mask_update;

%     updated = find(~cellfun(@isempty,list_added));
%     time_string = replace(char(datetime),':','-');
    fileName=sprintf('masks_processed(%d--%d).mat',find(list_processed,1),find(list_processed,1,'last'));
    if ~exist(folder,'dir')
        mkdir(folder);
    end
    while exist(fullfile(folder,fileName),'file')
        fileName=[fileName(1:end-4),'+','.mat'];
    end
%     gui.fileName = fileName;
    save(fullfile(folder,fileName), 'update_result');
    save(fullfile(folder,'masks_processed.mat'), 'update_result');
    close(gui.Window);
end

%% SaveTrace - saves labeling output file and closes GUI window
% function ChangeIoUth(~, ~)
%     str_thred_IoU = get(gui.IoUInput,'String');
%     thred_IoU = str2double(str_thred_IoU);
%     if isnan(thred_IoU)
%         auto_on = false;
%     else
%         if thred_IoU > 1
%             thred_IoU = 1;
%         elseif thred_IoU < 0
%             thred_IoU = 0;
%         end
%         auto_on = true;
%     end
%     set(gui.IoUInput,'String',num2str(thred_IoU));
% end

end