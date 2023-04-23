function GUI_refine(video,folder,masks,update_result)
% global video;
% global masks;
% global Lx;
% global Ly;
% global T;
% global N;
global thred_IoU;
global thred_IoU_split;
global thred_jaccard_inclass;
global thred_jaccard;
global auto_on;

% global variables calculated outside the GUI
global list_weight;
% global list_weight_trace;
% global list_weight_frame;
global traces_raw;
% global traces_out_exclude;
global traces_bg_exclude;
global area;
global r_bg_ratio;
global comx;
global comy;
global list_neighbors; 

% global variables for each neuron
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
global avg_frame; 
global confirmed_frames;
global classes;
global nearby_disk;
global current_thred;

% global variables related to display
global gui;
global video_sub_sort;
global mask_update_tile;
global mask_update_select;
global edges;
global sum_edges;
global active_ax;
global list_markers;
global list_frame;
global extra_button;
global new_contour;

% global variables summarizing results
global masks_update; 
global list_delete;
global list_added;
global ListStrings;
global list_IoU;
global list_avg_frame;
global list_mask_update; 

% global select_frames;
% global mask_update_select;
% global ncol; 
% global num_avg; 
% global nrow;

[Lx,Ly,T] = size(video);
[Lx,Ly,N] = size(masks);
ncol = 10;
num_avg = max(60, min(90, ceil(T*0.01)));
nrow = ceil(num_avg/ncol);
tile_size = [nrow, ncol];

thred_IoU_split = 0.5;
thred_jaccard_inclass = 0.4;
thred_jaccard = 0.7;

if exist('update_result','var')
    masks_update = update_result.masks_update;
    list_delete = update_result.list_delete;
    list_added = update_result.list_added;
    ListStrings = update_result.ListStrings;
    list_IoU = update_result.list_IoU;
    list_avg_frame = update_result.list_avg_frame;
    list_mask_update = update_result.list_mask_update;
    
    if ~size(masks_update) == size(masks)
        warning(['The size of "masks_update" is not consistent with the the size of the input "masks". ',...
            'Reset "masks_update" to an all zero array. ',...
            'Also reset other fields in "update_result". ']);
        masks_update = false(Lx,Ly,N);
        list_delete = false(1,N);
        list_added = {};
        ListStrings = num2cell(1:N);
        list_IoU = zeros(1,N);
        list_avg_frame = cell(1,N);
        list_mask_update = cell(1,N);
    end
else
    masks_update = false(Lx,Ly,N);
    list_delete = false(1,N);
    list_added = {};
    ListStrings = num2cell(1:N);
    list_IoU = zeros(1,N);
    list_avg_frame = cell(1,N);
    list_mask_update = cell(1,N);
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
color = color([2,3,4,6,7],:);
txtFntSz = 10;

maxImages = max(video,[],3);

nn = find(cellfun(@isempty,list_mask_update),1); %n = current neuron
if isempty(nn)
    nn = N;
end

%% Text messages
text_left1 = {'Blue: Original mask.    Red; Intensity isoheight in this frame.',...
    'Selected frames are sorted according to their weight.',...
    'The frame index "m_" and "_n" means this frame has the mn-th weight (starting from 00).'};
text_OX = {'O: Approved frame used for averaging; ',...
    'X: Disgarded frame not used for averaging. '};
text_012 = {'1: Frame used for the first neuron. This mask will take the place of the original mask. ',...
    '2: Frame used for the second neuron. This mask will be saved in an additional array. ',...
    '0: Disgarded frame not used for averaging. '};
text_center = {'Blue: Original mask',...
    'Red; Refined mask',...
    'Other colors: Original neighboring neurons'};
text_right_start = {'Do you want to update the mask of this neuron? You have the following options: ',...
    '1. Click "Yes, approve correction" to confirm update (use the red refined mask); ',...
    '2. Click "No, reject correction" to reject update (use the blue original mask); ',...
    '3. Click "Manual frame selection" if you want to manually remove some frames for averaing; ',...
    '4. Click "No neuron here" to delete this neuron form the neuron set; ',...
    ['5. Click "Multiple neurons here" if you find the selected frames include multiple neurons, ',...
    'but only one of them has been labeled; '],...
    '6. Click "Manually draw contour" to manually redraw the neuron contour.',...
    '7. Click "Save results" to save the current progress and exit the GUI.'};
text_right_manual_start = ['Click the first frame that you want to discard. ',...
    'All frames from this to the end will be selected to discard.',...
    'You can manually flip some selections in the next step.'];
text_right_manual_continue = ['Click the frames that you want to flip choice. ',...
    'Each clicked frame will be flipped from selected to discarded or vice versa. ',...
    'Press "Esc" to finish selection.'];
text_right_manual_confirm = {'Click "Confirm Selection" to use the red contour on the "Average Image" panel as the updated mask. ',...
    'Click "Change Selection" to continue editing the selected frames.'};
text_right_multi_continue = {['Click the frames that you want to change assignment. ',...
    'Each clicked frame will change assignment in cyclical from 0, 1, 2. ',...
    'Assign "1" to the neuron corresponding to the original mask. ',...
    'Assign "2" to the missing neuron. ',...
    'Press "Esc" to finish selection.'],...
    ['(This GUI only allows selecting two neurons (one extra). ',...
    'If there is more than one missing neuron, add them one by one. ',...
    'You can go back to this neuron by clicking "Backward Neuron" or double click the neuron index in the Neuron List, ',...
    'and click "Multiple neurons here" again to add additional neurons. ',...
    'Make sure always assign the same neuron to number "1".) ']};
text_right_multi_confirm = {'Click "No Multiple Neurons" if there is not a missing neuron. ',...
    'Click "Confirm Selection" to use the red contours on the "Average Image" panel as the updated masks. ',...
    'Click "Change Selection" to continue editing the selected frames.'};
text_right_no_confirm = {'Click "Confirm No Neuron" to delete the neuron. ',...
    'Click "Manually Draw Neuron" to manually draw the mask. ',...
    'Click "No, reject correction" on the right panel to return to the original blue mask'};
text_right_draw_select = {'Click "Draw on Peak SNR" to draw on the top image panel. ',...
    'Click "Draw on Average Frame" to draw on the bottom image panel.'};
text_right_draw = ['Use mouse to draw the contour on your selected image panel. '];
text_right_draw_confirm = {'Click "Confirm Drawing" to confirm the manual drawing. ',...
    'Click "Redo Drawing" to redraw.'};

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
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Confirm Selection', ...
        'FontSize', txtFntSz);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Change Selection', ...
        'FontSize', txtFntSz);
    button3_position = [Frames_position(1)+Frames_position(3)-600, 10, 190, 50];
    gui.Single = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button3_position, ...
        'Style', 'PushButton', ...
        'String', 'Single Neuron', ...
        'FontSize', txtFntSz, ...
        'CallBack', @Single);
    ContoursButton_position = [Frames_position(1), 10, 250, 50];
    gui.ShowContoursButton = uicontrol(...
        'Style', 'ToggleButton', ...
        'Parent', gui.FramesPanel, ...
        'Position', ContoursButton_position, ...
        'String', 'Hide contours in each frame', ...
        'FontSize', txtFntSz, ...
        'Value', 1, ...
        'CallBack', @ShowContours);
    PlayButton_position = [Frames_position(1)+260, 10, 150, 50];
    gui.PlayTransientButton = uicontrol(...
        'Style', 'ToggleButton', ...
        'Parent', gui.FramesPanel, ...
        'Position', PlayButton_position, ...
        'String', 'Paly a transient', ...
        'FontSize', txtFntSz, ...
        'Value', 1, ...
        'CallBack', @PlayTransient);

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
        'String', '');
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
    gui.ClimPanelRight = uix.VBoxFlex(...
        'Parent', gui.ClimPanel, ...
        'Padding', 3);
    gui.HigherClim = uicontrol(...
        'Parent', gui.ClimPanelRight, ...
        'Style', 'PushButton', ...
        'String', 'Larger colorbar range', ...
        'FontSize', txtFntSz, ...
        'CallBack', @HigherClim);
    gui.LowerClim = uicontrol(...
        'Parent', gui.ClimPanelRight, ...
        'Style', 'PushButton', ...
        'String', 'Smaller colorbar range', ...
        'FontSize', txtFntSz, ...
        'CallBack', @LowerClim);
    set(gui.CenterLayout, 'Heights', [- 4.5, - 5, - .5, - 1]);

    gui.AvgTabGroup = uitabgroup(...
        'Parent', gui.AvgPanel);
%     delete(gui.Confirm);
%     delete(gui.Change)
%     delete(gui.Single)
%     delete(gui.AvgTabGroup)
    gui.masks_update_tile = gui.AvgTabGroup;
    delete(gui.masks_update_tile);
    gui.mask_update = gui.AvgTabGroup;
    delete(gui.mask_update);
    gui.hFH = gui.AvgTabGroup;
    delete(gui.hFH);
    gui.FinishInfo = gui.AvgTabGroup;
    delete(gui.FinishInfo);

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
    gui.ConfirmButton = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'Yes, approve correction', ...
        'FontSize', txtFntSz, ...
        'CallBack', @Confirm);
    gui.RevertButton = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'No, reject correction', ...
        'FontSize', txtFntSz, ...
        'CallBack', @Reject);
    gui.ManualButton = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'Manual frame selection', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ManualSelection);
    gui.NoNeuronButton = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'No neuron here', ...
        'FontSize', txtFntSz, ...
        'CallBack', @NoNeurons);
    gui.MultipleButton = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'Multiple neurons here', ...
        'FontSize', txtFntSz, ...
        'CallBack', @MultipleNeurons);
    gui.ManualDrawButton = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.ControlPanel, ...
        'String', 'Manually draw contour', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ManualDrawSelect);
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
    gui.IoUgt = uicontrol(...
        'Parent', gui.IoULabel, ...
        'Style', 'text', ...
        'String', 'IoU >', ...
        'FontSize', txtFntSz, ...
        'HorizontalAlignment', 'right');
    gui.IoUInput = uicontrol(...
        'Parent', gui.IoULabel, ...
        'Style', 'edit', ...
        'String', '0.7', ...
        'FontSize', txtFntSz, ...
        'InnerPosition', [5,5,100,15], ...
        'HorizontalAlignment', 'left');
    gui.ChangeIoUth = uicontrol(...
        'Style', 'PushButton', ...
        'Parent', gui.IoULabel, ...
        'String', 'Change', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ChangeIoUth);

    set(gui.RightLayout, 'Heights', [- 3, - 3, - 3, - 1]);
    IoUgt_Position = gui.IoUgt.Position;
    IoUgt_Position(2) = IoUgt_Position(2) + IoUgt_Position(4) - txtFntSz*2;
    IoUgt_Position(4) = txtFntSz*2;
    IoUInput_Position = gui.IoUInput.Position;
    IoUInput_Position(2) = IoUInput_Position(2) + IoUInput_Position(4) - txtFntSz*2;
    IoUInput_Position(4) = txtFntSz*2;
    ChangeIoUth_Position = gui.ChangeIoUth.Position;
    ChangeIoUth_Position(2) = ChangeIoUth_Position(2) + ChangeIoUth_Position(4) - txtFntSz*2;
    ChangeIoUth_Position(4) = txtFntSz*2;
    set(gui.IoUgt,'Position',IoUgt_Position);
    set(gui.IoUInput,'Position',IoUInput_Position);
    set(gui.ChangeIoUth,'Position',ChangeIoUth_Position);
    
    extra_button = 0;
    str_thred_IoU = inputdlg('Minimum IoU to automatically approve correction','IoU threshold',1,{'1'});
    thred_IoU = str2double(str_thred_IoU{1});
%     ChangeIoUth();
%     while isnan(thred_IoU) 
%         str_thred_IoU = inputdlg('Minimum IoU to automatically approve correction','IoU threshold',1,{'0.7'});
%         thred_IoU = str2double(str_thred_IoU{1});
%     end
    if isnan(thred_IoU)
        auto_on = false;
    else
        if thred_IoU > 1
            thred_IoU = 1;
        elseif thred_IoU < 0
            thred_IoU = 0;
        end
        auto_on = true;
    end
    set(gui.IoUInput,'String',thred_IoU);
end

%% updateInterface - update GUI for new trace
function updateInterface()
    delete(gui.Confirm);
    delete(gui.Change)
    delete(gui.Single)
    delete(gui.AvgTabGroup)
    delete(gui.FinishInfo);
    extra_button = 0;
    
    %% Update the average image
    mask = masks(:,:,nn);
    r_bg = sqrt(mean(area)/pi)*r_bg_ratio;
    r_bg_ext = r_bg*3/2;
    xmin = max(1,round(comx(nn)-r_bg_ext));
    xmax = min(Lx,round(comx(nn)+r_bg_ext));
    ymin = max(1,round(comy(nn)-r_bg_ext));
    ymax = min(Ly,round(comy(nn)+r_bg_ext));
    xrange = xmin:xmax;
    yrange = ymin:ymax;
    mask_sub = mask(xrange,yrange);
    video_sub = video(xrange,yrange,:);
    [Lxm, Lym] = size(mask_sub);
    
    [yy,xx] = meshgrid(yrange,xrange);
    nearby_disk = ((xx-comx(nn)).^2 + (yy-comy(nn)).^2) < (r_bg)^2;
    weight = list_weight{nn};
    
    select_frames = find(weight>0);
    [weight_sort,select_frames_order] = sort(weight(select_frames),'descend');
    select_frames_sort = select_frames(select_frames_order);
    updateAverage();
    updateTiles(max(avg_frame(mask_sub)));
    
    %% Update summary image
    axes(gui.SummaryAxes);
    cla(gui.SummaryAxes);
    imagesc(gui.SummaryAxes,maxImages(xrange,yrange));
    colormap(gui.SummaryAxes, 'gray');
    colorbar(gui.SummaryAxes);
    axis(gui.SummaryAxes,'image');
    hold(gui.SummaryAxes, 'on');
    neighbors = list_neighbors{nn};
    for kk = 1:length(neighbors)
        contour(gui.SummaryAxes,masks(xrange,yrange,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
    end
    contour(gui.SummaryAxes,mask_sub,'b');
    contour(gui.SummaryAxes,mask_update,'r');
    image_green = zeros(Lxm,Lym,3,'uint8');
    image_green(:,:,2)=255;
    edge_others = sum_edges(xrange,yrange)-sum(edges(xrange,yrange,[nn,neighbors]),3);
    imagesc(gui.SummaryAxes,image_green,'alphadata',0.5*edge_others);
    title(gui.SummaryAxes,'Peak SNR (entire video)')
    
    %% Update the button group
    set(gui.InfoText, 'String', text_right_start);
    
    %% Try to cluster the frames to two neurons
    if n_select == 0
        classes = [];
    elseif n_select == 1
        classes = 1;
        if auto_on && list_IoU(nn) > thred_IoU
            Confirm()
        end
    else        
        select_frames_sort_full = select_frames_sort;
        mask_update_select_2 = reshape(mask_update_select,Lxm*Lym,n_select)';
        dist =  pdist(double(mask_update_select_2),'jaccard'); 
        dist(dist>thred_jaccard) = nan;
        tree = linkage(dist,'average');
        classes = cluster(tree,'MaxClust',2);
    %     classes = clusterdata(double(mask_update_select_2),'MaxClust',2,'Distance','jaccard','Linkage','average');
    %     video_sub_sort_2 = reshape(video_sub_sort,Lxm*Lym,n_select)';
    %     classes = clusterdata(video_sub_sort_2,'Distance','cosine','MaxClust',2);
        if sum(classes == 1) < sum(classes == 2)
            classes = 3 - classes;
        end
    %     classes = ones(1,n_select);

        dist_2 = squareform(dist);
        dist_2_2 = dist_2(classes == 2,classes == 2) + eye(sum(classes == 2));
        min_dist_2_2 = min(dist_2_2,[],'all');
        if min_dist_2_2 > thred_jaccard_inclass % only one neuron
            if auto_on && list_IoU(nn) > thred_IoU
                Confirm()
            end
        else
            avg_mask_1 = mean(mask_update_select_2(classes == 1,:),1);
            avg_mask_1 = avg_mask_1 > 0.5*max(avg_mask_1);
            avg_mask_2 = mean(mask_update_select_2(classes == 2,:),1);
            avg_mask_2 = avg_mask_2 > 0.5*max(avg_mask_2);
            mask_sub_2 = reshape(mask_sub,1,Lxm*Lym);
            IoU_12 = sum(avg_mask_1 & avg_mask_2) / sum(avg_mask_1 | avg_mask_2);
            IoU_1 = sum(avg_mask_1 & mask_sub_2) / sum(avg_mask_1 | mask_sub_2);
            IoU_2 = sum(avg_mask_2 & mask_sub_2) / sum(avg_mask_2 | mask_sub_2);
            if IoU_1 < IoU_2
                classes = 3 - classes;
            end

            %% If IoU is large enough, automatically accept the updated mask
            if IoU_12 < thred_IoU_split
                MultipleNeurons();
            elseif auto_on && list_IoU(nn) > thred_IoU
                Confirm()
            end
        end
    end
end

%% Update selected frames
function updateTiles(max_inten)
%     if IoU<0.6 % || length(select_frames)<0.01*T
    n_select = length(select_frames_sort);
    video_sub_sort = video_sub(:,:,select_frames_sort);
    images_tile = imtile(video_sub_sort, 'GridSize', tile_size);
    mask_sub_tile = repmat(mask_sub,tile_size);
    q = 1-mean(mask_sub,'all');
    mask_update_select = false(Lxm,Lym,n_select);
    for kk = 1:n_select
        frame = video_sub_sort(:,:,kk);
        frame(~nearby_disk) = 0;
        thred_inten = quantile(frame, q, 'all');
        mask_update_select(:,:,kk) = threshold_frame(frame, thred_inten, area(nn), mask_sub);
%         mask_update_select(:,:,kk) = frame > thred_inten;
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
    if ~exist('max_inten','var')
        max_inten = prctile(video_sub_sort, 99,'all');
    end
    imagesc(gui.FramesAxes, images_tile,[-1,max_inten]);
    colormap(gui.FramesAxes,'gray')
    colorbar(gui.FramesAxes);
    axis(gui.FramesAxes,'image');
    hold(gui.FramesAxes,'on');
%         image_green = zeros(size(images_tile,1),size(images_tile,2),3,'uint8');
%         image_green(:,:,2)=255;
%         imagesc(image_green,'alphadata',0.5*edge_others_tile);
    contour(gui.FramesAxes,mask_sub_tile,'b');
    if gui.ShowContoursButton.Value
        [~,gui.masks_update_tile] = contour(gui.FramesAxes,mask_update_tile,'r');
    end
    list_frame = cell(1,n_select);
    for kk = 1:n_select
        [yn,xn] = ind2sub([ncol,nrow],kk);
        xx = (xn)*Lxm-6;
        yy = (yn-1)*Lym+4;
        list_frame{kk} = text(gui.FramesAxes,yy,xx,num2str(select_frames_sort(kk)),'FontSize',10,'Color','y');
    end
    title(gui.FramesAxes,['Neuron ',num2str(nn)]);
    set(gui.FramesAxes,...
        'XTick',round(Lym*(0.5:ncol-0.5)), 'YTick',round(Lxm*(0.5:nrow-0.5)), ...
        'XTickLabel',arrayfun(@(x) ['_',num2str(x)], 0:ncol-1, 'UniformOutput',false),...
        'YTickLabel',arrayfun(@(x) [num2str(x),'_'], 0:nrow-1, 'UniformOutput',false),...
        'TickLabelInterpreter','None');
    set(gui.InfoTextLeft2, 'String', '');
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

%%
function updateAverage()
    if isempty(select_frames_sort)
        avg_frame = 0*mask_sub;
        mask_update = logical(0*mask_sub);
        current_thred = 0;
    else
        avg_frame = sum(video_sub(:,:,select_frames_sort).*reshape(weight(select_frames_sort),1,1,[]),3)./sum(weight(select_frames_sort));
        list_avg_frame{nn} = avg_frame;
        avg_frame_use = avg_frame;
        avg_frame_use(~nearby_disk) = 0;
        thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all'), 'all');
        [mask_update, ~, current_thred] = threshold_frame(avg_frame_use, thred_inten, area(nn), mask_sub);
    end
%     mask_update = avg_frame_use > thred_inten;
%     [L,nL] = bwlabel(mask_update,4);
%     if nL>1
%         list_area_L = zeros(nL,1);
%         for kk = 1:nL
%             list_area_L(kk) = sum(L==kk,'all');
%         end
%         [max_area,iL] = max(list_area_L);
% %             for test = 1:3
%         while max_area < area(nn)
%             thred_inten = thred_inten - 0.1;
% %                 thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all')/max_area*area(nn), 'all');
%             mask_update = avg_frame_use > thred_inten;
%             [L,nL] = bwlabel(mask_update,4);
%             list_area_L = zeros(nL,1);
%             for kk = 1:nL
%                 list_area_L(kk) = sum(L==kk,'all');
%             end
%             [max_area,iL] = max(list_area_L);
%         end
%         mask_update = (L==iL);
%     end
% 
%     [L0,nL0] = bwlabel(~mask_update,4);
%     if nL0>1
%         list_area_L0 = zeros(nL0,1);
%         for kk = 1:nL0
%             list_area_L0(kk) = sum(L0==kk,'all');
%         end
%         [max_area,iL0] = max(list_area_L0);
%         mask_update = (L0 ~= iL0);
%     end
    %%
    area_i = sum(mask_update & mask_sub,'all');
    area_u = sum(mask_update | mask_sub,'all');
    IoU = area_i/area_u;
    list_IoU(nn) = IoU;
    
    %% Update average frames
    axes(gui.AvgAxes);
    cla(gui.AvgAxes);
    imagesc(gui.AvgAxes,avg_frame);
    colormap(gui.AvgAxes,'gray');
    colorbar(gui.AvgAxes);
    axis(gui.AvgAxes,'image');
    set(gui.AvgAxes,'CLimMode','auto');
    Clim = get(gui.AvgAxes,'Clim');
    if Clim(2)>8
        Clim(2)=8;
        set(gui.AvgAxes,'Clim',Clim);
    end
    hold(gui.AvgAxes,'on');
    neighbors = list_neighbors{nn};
    for kk = 1:length(neighbors)
        contour(gui.AvgAxes,masks(xrange,yrange,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
    end
    contour(gui.AvgAxes,mask_sub,'b');
    [~, new_contour] = contour(gui.AvgAxes,mask_update,'r');
    image_green = zeros(Lxm,Lym,3,'uint8');
    image_green(:,:,2)=255;
    edge_others = sum_edges(xrange,yrange)-sum(edges(xrange,yrange,[nn,neighbors]),3);
    imagesc(gui.AvgAxes,image_green,'alphadata',0.5*edge_others);
    title(gui.AvgAxes,sprintf('Average frame, IoU = %0.2f',IoU))
end 

%%
function updateAverageTab(t)
    if t == 1
        avg_frame{t} = sum(video_sub(:,:,select_frames_sort{t}).*reshape(weight(select_frames_sort{t}),1,1,[]),3)./sum(weight(select_frames_sort{t}));
    else % weighted average only applies to the home neuron. 
        avg_frame{t} = mean(video_sub(:,:,select_frames_sort{t}),3);
    end
    avg_frame_use = avg_frame{t};
    avg_frame_use(~nearby_disk) = 0;
    thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all'), 'all');
    [mask_update{t}, ~, current_thred{t}] = threshold_frame(avg_frame_use, thred_inten, area(nn), mask_sub);
%     mask_update{t} = avg_frame_use > thred_inten;
%     [L,nL] = bwlabel(mask_update{t},4);
%     if nL>1
%         list_area_L = zeros(nL,1);
%         for kk = 1:nL
%             list_area_L(kk) = sum(L==kk,'all');
%         end
%         [max_area,iL] = max(list_area_L);
% %             for test = 1:3
%         while max_area < area(nn)
%             thred_inten = thred_inten - 0.1;
% %                 thred_inten = quantile(avg_frame_use, 1-mean(mask_sub,'all')/max_area*area(nn), 'all');
%             mask_update{t} = avg_frame_use > thred_inten;
%             [L,nL] = bwlabel(mask_update{t},4);
%             list_area_L = zeros(nL,1);
%             for kk = 1:nL
%                 list_area_L(kk) = sum(L==kk,'all');
%             end
%             [max_area,iL] = max(list_area_L);
%         end
%         mask_update{t} = (L==iL);
%     end
% 
%     [L0,nL0] = bwlabel(~mask_update{t},4);
%     if nL0>1
%         list_area_L0 = zeros(nL0,1);
%         for kk = 1:nL0
%             list_area_L0(kk) = sum(L0==kk,'all');
%         end
%         [max_area,iL0] = max(list_area_L0);
%         mask_update{t} = (L0 ~= iL0);
%     end
    area_i = sum(mask_update{t} & mask_sub,'all');
    area_u = sum(mask_update{t} | mask_sub,'all');
    IoU = area_i/area_u;
    if t == 1
        list_IoU(nn) = IoU;
    end
    
    %% Update average frames
    axes(gui.AvgTabAxes{t});
    cla(gui.AvgTabAxes{t});
    imagesc(gui.AvgTabAxes{t},avg_frame{t});
    colormap(gui.AvgTabAxes{t},'gray');
    colorbar(gui.AvgTabAxes{t});
    axis(gui.AvgTabAxes{t},'image');
    Clim = get(gui.AvgTabAxes{t},'Clim');
    if Clim(2)>8
        Clim(2)=8;
        set(gui.AvgTabAxes{t},'Clim',Clim);
    end
    hold(gui.AvgTabAxes{t},'on');
    neighbors = list_neighbors{nn};
    for kk = 1:length(neighbors)
        contour(gui.AvgTabAxes{t},masks(xrange,yrange,neighbors(kk)),'Color',color(mod(kk,size(color,1))+1,:));
    end
    contour(gui.AvgTabAxes{t},mask_sub,'b');
    [~, new_contour{t}] = contour(gui.AvgTabAxes{t},mask_update{t},'r');
    image_green = zeros(Lxm,Lym,3,'uint8');
    image_green(:,:,2)=255;
    edge_others = sum_edges(xrange,yrange)-sum(edges(xrange,yrange,[nn,neighbors]),3);
    imagesc(gui.AvgTabAxes{t},image_green,'alphadata',0.5*edge_others);
    title(gui.AvgTabAxes{t},sprintf('Average frame, IoU = %0.2f',IoU))
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
function Confirm(~, ~)
    list_mask_update{nn} = mask_update;
    masks_update(xrange,yrange,nn) = mask_update;
    ListStrings{nn} = [num2str(nn),' Updated'];
    
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%% NoSpike - Check if user is looking for spikes, print to output file,
%and play video for next spike
function Reject(~, ~)
    list_mask_update{nn} = mask_sub;
    masks_update(xrange,yrange,nn) = mask_sub;
    ListStrings{nn} = [num2str(nn),' Rejected'];
    
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

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
function ManualSelection(~,~)
    delete(gui.Confirm)
    delete(gui.Change)
    delete(gui.Single)
    delete(gui.AvgTabGroup)
    set(gui.InfoText, 'String', text_right_manual_start);
    ind = select_frame();
    if isnan(ind)
        set(gui.InfoText, 'String', text_right_start);
    else
        confirmed_frames = true(1,n_select);
        confirmed_frames(ind:end) = false;
        list_markers = cell(1,n_select);
        for n = 1:n_select
            [yn,xn] = ind2sub([ncol,nrow],n);
            xx = (xn-1)*Lxm+4;
            yy = (yn-1)*Lym+4;
            if confirmed_frames(n)
                list_markers{n} = plot(gui.FramesAxes,yy,xx,'yo','MarkerSize',10);
            else
                list_markers{n} = plot(gui.FramesAxes,yy,xx,'yx','MarkerSize',10);
            end
        end
        
        select_frames_sort_full = select_frames_sort;
        select_frames_sort = select_frames_sort_full(confirmed_frames);
        updateAverage();
        Frames_position = gui.FramesPanel.Position;
        button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
        gui.Confirm = uicontrol(...
            'Parent', gui.FramesPanel, ...
            'Position', button1_position, ...
            'Style', 'PushButton', ...
            'String', 'Confirm Selection', ...
            'FontSize', txtFntSz, ...
            'CallBack', @ConfirmSelect);
        button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
        gui.Change = uicontrol(...
            'Parent', gui.FramesPanel, ...
            'Position', button2_position, ...
            'Style', 'PushButton', ...
            'String', 'Change Selection', ...
            'FontSize', txtFntSz, ...
            'CallBack', @ChangeSelect);
        extra_button = 3;
        set(gui.InfoText, 'String', text_right_manual_confirm);
%         ChangeSelect();
    end
end

%%
function PlayTransient(~,~)
%     delete(gui.Play);
    Frames_position = gui.FramesPanel.Position;
    esc_position = [Frames_position(1)+Frames_position(3)-400, 10, 390, 50];
    gui.FinishInfo = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', esc_position, ...
        'Style', 'text', ...
        'String', 'Click the transient to play', ...
        'FontSize', 20);
    extra_button = 0;

%     set(gui.InfoText, 'String', text_right_manual_continue);
%     set(gui.InfoTextLeft2, 'String', text_OX);
    ind = select_frame();
    kk = select_frames_sort_full(ind);
    transient_play = video_sub(:,:,max(1,kk-20):min(T,kk+40));
    figure(110);
    imshow3D(transient_play,get(gui.FramesAxes,'Clim'),min(kk,21));
    delete(gui.FinishInfo);
end

%%
function ChangeSelect(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    Frames_position = gui.FramesPanel.Position;
    esc_position = [Frames_position(1)+Frames_position(3)-400, 10, 390, 50];
    gui.FinishInfo = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', esc_position, ...
        'Style', 'text', ...
        'String', 'Press "Esc" to finish', ...
        'FontSize', 20);
    extra_button = 0;

    set(gui.InfoText, 'String', text_right_manual_continue);
    set(gui.InfoTextLeft2, 'String', text_OX);
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

    select_frames_sort = select_frames_sort_full(confirmed_frames);
    updateAverage();
    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Confirm Selection', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ConfirmSelect);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Change Selection', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ChangeSelect);
    extra_button = 3;
    set(gui.InfoText, 'String', text_right_manual_confirm);
end

%%
function ConfirmSelect(~,~)
    clear select_frames_sort_full;
    delete(gui.Confirm);
    delete(gui.Change)
    extra_button = 0;
    
    list_mask_update{nn} = mask_update;
    masks_update(xrange,yrange,nn) = mask_update;
    ListStrings{nn} = [num2str(nn),' Curated'];
    
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%%
function NoNeurons(~,~)
    select_frames_sort_full = select_frames_sort;
    trace_bgsubs = traces_raw(nn,:) - traces_bg_exclude(nn,:);
    [~, locs, ~, proms] = findpeaks(trace_bgsubs);
    [~,select_frames_sort] = sort(proms,'descend');
    select_frames_sort = locs(select_frames_sort);
%     [trace_bgsubs_sort,select_frames_sort] = sort(trace_bgsubs,'descend');
    if length(select_frames_sort) > num_avg
        select_frames_sort = select_frames_sort(1:num_avg);
    end
    updateTiles();
    
    delete(gui.Confirm)
    delete(gui.Change)
    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Confirm No Neuron', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ConfirmNoNeuron);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Manually Draw Neuron', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ManualDrawSelect);
    extra_button = 4;
    set(gui.InfoText, 'String', text_right_no_confirm);
end

%%
function ConfirmNoNeuron(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    extra_button = 0;
    
    list_mask_update{nn} = false(Lxm,Lym);
    masks_update(xrange,yrange,nn) = false(Lxm,Lym);
    list_delete(nn) = true; 
    ListStrings{nn} = [num2str(nn),' Deleted'];
    
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%%
function MultipleNeurons(~,~)
    list_markers = cell(1,n_select);
    for n = 1:n_select
        [yn,xn] = ind2sub([ncol,nrow],n);
        xx = (xn-1)*Lxm+4;
        yy = (yn-1)*Lym+4;
        list_markers{n} = text(gui.FramesAxes,yy,xx,...
            num2str(classes(n)),'Color','y','FontSize',14);
    end
    
    delete(gui.AvgTabGroup)
    gui.AvgTabGroup = uitabgroup(...
        'Parent', gui.AvgPanel);
    gui.AvgTab = cell(1,2);
    gui.AvgTabAxes = cell(1,2);
    gui.AvgTab{1} = uitab(gui.AvgTabGroup, ...
        'Title', 'Averaged Image 1');
    gui.AvgTabAxes{1} = axes(...
        'Parent', gui.AvgTab{1});
    gui.AvgTab{2} = uitab(gui.AvgTabGroup, ...
        'Title', 'Averaged Image 2');
    gui.AvgTabAxes{2} = axes(...
        'Parent', gui.AvgTab{2});
    set(gui.AvgTabGroup,'SelectedTab',gui.AvgTab{2});
    
%     ChangeSelectMulti();
    select_frames_sort = {select_frames_sort_full(classes==1),...
        select_frames_sort_full(classes==2)};
    [avg_frame, mask_update, new_contour, current_thred] = deal(cell(1,2));
    updateAverageTab(1);
    updateAverageTab(2);
    list_avg_frame{nn} = avg_frame;

    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Confirm Multiple Neurons', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ConfirmSelectMulti);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Change Selection', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ChangeSelectMulti);
    button3_position = [Frames_position(1)+Frames_position(3)-600, 10, 190, 50];
    gui.Single = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button3_position, ...
        'Style', 'PushButton', ...
        'String', 'Single Neuron', ...
        'FontSize', txtFntSz, ...
        'CallBack', @Single);
    extra_button = 5;
    set(gui.InfoText, 'String', text_right_multi_confirm);
end

%%
function ChangeSelectMulti(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    delete(gui.Single)
    Frames_position = gui.FramesPanel.Position;
    button3_position = [Frames_position(1)+Frames_position(3)-400, 10, 390, 50];
    gui.FinishInfo = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button3_position, ...
        'Style', 'text', ...
        'String', 'Press "Esc" to finish', ...
        'FontSize', 20);
    extra_button = 0;

    set(gui.InfoText, 'String', text_right_multi_continue);
    set(gui.InfoTextLeft2, 'String', text_012);
    ind = select_frame();
    while ~isnan(ind)
        if ind <= n_select
            classes(ind) = mod(classes(ind)+1,3);
            delete(list_markers{ind})
            [yn,xn] = ind2sub([ncol,nrow],ind);
            xx = (xn-1)*Lxm+4;
            yy = (yn-1)*Lym+4;
            list_markers{ind} = text(gui.FramesAxes,yy,xx,...
                num2str(classes(ind)),'Color','y','FontSize',14);
        end
        ind = select_frame();
    end
    delete(gui.FinishInfo);

    select_frames_sort = {select_frames_sort_full(classes==1),...
        select_frames_sort_full(classes==2)};
    [avg_frame,mask_update] = deal(cell(1,2));
    updateAverageTab(1);
    updateAverageTab(2);
    list_avg_frame{nn} = avg_frame;

    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Confirm Selection', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ConfirmSelectMulti);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Change Selection', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ChangeSelectMulti);
    extra_button = 5;
    set(gui.InfoText, 'String', text_right_multi_confirm);
end

%%
function ConfirmSelectMulti(~,~)
    clear select_frames_sort_full;
    delete(gui.Confirm);
    delete(gui.Change)
    delete(gui.Single)
    delete(gui.AvgTabGroup)
    extra_button = 0;

    list_mask_update{nn} = mask_update{1};
    masks_update(xrange,yrange,nn) = mask_update{1};
    mask_new = false(Lx,Ly);
    mask_new(xrange,yrange) = mask_update{2};
    list_added{end+1} = mask_new;
    ListStrings{nn} = [num2str(nn),' Added'];
    
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%%
function Single(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    delete(gui.Single)
    delete(gui.AvgTabGroup)
    extra_button = 0;
    for n = 1:n_select
        delete(list_markers{n});
    end
    select_frames_sort = select_frames_sort_full;
    updateAverage()
%     mask_update = mask_update_1;
%     avg_frame = avg_frame_1;
    set(gui.InfoText, 'String', text_right_start);
    if auto_on && list_IoU(nn) > thred_IoU
        Confirm()
    end
end

%%
function ManualDraw(~,~)
    set(gui.InfoText, 'String', text_right_draw);
    delete(gui.Confirm);
    delete(gui.Change)
    delete(gui.Single)
    delete(gui.AvgTabGroup)
    extra_button = 0;

%     if isfield(gui,'mask_update')
        delete(gui.mask_update);
%     end
%     if isfield(gui,'hFH')
        delete(gui.hFH);
%     end
    
    gui.hFH = drawfreehand(active_ax);
    if ~isempty(gui.hFH) && isvalid(gui.hFH)
        imH = imhandles(active_ax);
        mask_update = gui.hFH.createMask(imH(1));
        if nnz(mask_update)
            [~, gui.mask_update] = contour(active_ax,mask_update,'Color','m');
        end
    end
    
    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-400, 10, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Confirm Drawing', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ConfirmManualDraw);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Redo Drawing', ...
        'FontSize', txtFntSz, ...
        'CallBack', @ManualDraw);
    extra_button = 6;
    set(gui.InfoText, 'String', text_right_draw_confirm);
end

%%
function ManualDrawSelect(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    delete(gui.Single)
    delete(gui.AvgTabGroup)
    extra_button = 0;
    
    Frames_position = gui.FramesPanel.Position;
    button1_position = [Frames_position(1)+Frames_position(3)-200, 70, 190, 50];
    gui.Confirm = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button1_position, ...
        'Style', 'PushButton', ...
        'String', 'Draw on Peak SNR', ...
        'FontSize', txtFntSz, ...
        'CallBack', @DrawTop);
    button2_position = [Frames_position(1)+Frames_position(3)-200, 10, 190, 50];
    gui.Change = uicontrol(...
        'Parent', gui.FramesPanel, ...
        'Position', button2_position, ...
        'Style', 'PushButton', ...
        'String', 'Draw on Average Frame', ...
        'FontSize', txtFntSz, ...
        'CallBack', @DrawBottom);
    extra_button = 6;
    set(gui.InfoText, 'String', text_right_draw_select);
end

%%
function DrawTop(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    extra_button = 0;
    active_ax = gui.SummaryAxes;
    ManualDraw();
end

%%
function DrawBottom(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    extra_button = 0;
    active_ax = gui.AvgAxes;
    ManualDraw();
end

%%
function ConfirmManualDraw(~,~)
    delete(gui.Confirm);
    delete(gui.Change)
    extra_button = 0;

    list_mask_update{nn} = mask_update;
    masks_update(xrange,yrange,nn) = mask_update;
    ListStrings{nn} = [num2str(nn),' Redrawn'];
    
    set(gui.ListBox, 'String', ListStrings);
    Forward();
end

%% SaveTrace - saves labeling output file and closes GUI window
function SaveMasks(~, ~)
    update_result.masks_update = masks_update;
    update_result.list_delete = list_delete;
    update_result.list_added = list_added;
    update_result.ListStrings = ListStrings;
    update_result.list_IoU = list_IoU;
    update_result.list_avg_frame = list_avg_frame;
    update_result.list_mask_update = list_mask_update;

    updated = find(~cellfun(@isempty,list_mask_update));
%     time_string = replace(char(datetime),':','-');
    if ~isempty(updated)
        fileName=sprintf('masks_update(%d--%d)-%d+%d.mat',updated(1),updated(end),sum(list_delete),length(list_added));
    else
        fileName=sprintf('masks_update-%d+%d.mat',sum(list_delete),length(list_added));
    end
    if ~exist(folder,'dir')
        mkdir(folder);
    end
    while exist(fullfile(folder,fileName),'file')
        fileName=[fileName(1:end-4),'+','.mat'];
    end
    save(fullfile(folder,'masks_update.mat'), 'update_result');
    save(fullfile(folder,fileName), 'update_result');
    close(gui.Window);
end

%% SaveTrace - saves labeling output file and closes GUI window
function ChangeIoUth(~, ~)
    str_thred_IoU = get(gui.IoUInput,'String');
    thred_IoU = str2double(str_thred_IoU);
    if isnan(thred_IoU)
        auto_on = false;
    else
        if thred_IoU > 1
            thred_IoU = 1;
        elseif thred_IoU < 0
            thred_IoU = 0;
        end
        auto_on = true;
    end
    set(gui.IoUInput,'String',num2str(thred_IoU));
end

%%
function LowerThreshold(~,~)
    if iscell(avg_frame)
        t = str2double(gui.AvgTabGroup.SelectedTab.Title(end));
        avg_frame_use = avg_frame{t};
        avg_frame_use(~nearby_disk) = 0;
        [mask_update{t}, ~, current_thred{t}] = threshold_frame(avg_frame_use, current_thred{t}*0.9, [], mask_sub);
        delete(new_contour{t});
        [~, new_contour{t}] = contour(gui.AvgTabAxes{t},mask_update{t},'r');
    else
        avg_frame_use = avg_frame;
        avg_frame_use(~nearby_disk) = 0;
        [mask_update, ~, current_thred] = threshold_frame(avg_frame_use, current_thred*0.9, [], mask_sub);
        delete(new_contour);
        [~, new_contour] = contour(gui.AvgAxes,mask_update,'r');
    end
end

%%
function HigherThreshold(~,~)
    if iscell(avg_frame)
        t = str2double(gui.AvgTabGroup.SelectedTab.Title(end));
        avg_frame_use = avg_frame{t};
        avg_frame_use(~nearby_disk) = 0;
        [mask_update{t}, ~, current_thred{t}] = threshold_frame(avg_frame_use, current_thred{t}*1.1, [], mask_sub);
        delete(new_contour{t});
        [~, new_contour{t}] = contour(gui.AvgTabAxes{t},mask_update{t},'r');
    else
        avg_frame_use = avg_frame;
        avg_frame_use(~nearby_disk) = 0;
        [mask_update, ~, current_thred] = threshold_frame(avg_frame_use, current_thred*1.1, [], mask_sub);
        delete(new_contour);
        [~, new_contour] = contour(gui.AvgAxes,mask_update,'r');
    end
end

%%
function HigherClim(~,~)
    if iscell(avg_frame)
        t = str2double(gui.AvgTabGroup.SelectedTab.Title(end));
        Clim = get(gui.AvgTabAxes{t},'Clim');
        Clim(2) = Clim(2) + 1;
        set(gui.AvgTabAxes{t},'Clim',Clim);
    else
        Clim = get(gui.AvgAxes,'Clim');
        Clim(2) = Clim(2) + 1;
        set(gui.AvgAxes,'Clim',Clim);
    end
%     set(gui.FramesAxes,'Clim',Clim);
end

%%
function LowerClim(~,~)
    if iscell(avg_frame)
        t = str2double(gui.AvgTabGroup.SelectedTab.Title(end));
        Clim = get(gui.AvgTabAxes{t},'Clim');
        Clim(2) = Clim(2) - 1;
        set(gui.AvgTabAxes{t},'Clim',Clim);
    else
        Clim = get(gui.AvgAxes,'Clim');
        Clim(2) = Clim(2) - 1;
        set(gui.AvgAxes,'Clim',Clim);
    end
%     set(gui.FramesAxes,'Clim',Clim);
end

end