%
% NeuronFilter - Temporal Labeling GUI for calcium imaging videos of active neurons
% Last updated 8/29/2019 by Agnim Agarwal
%

function NeuronFilter(vid, Mask, markings, trace) % markings variable =  for external markings OR 0 for no file input
% vid is the input video
% Mask is the mask of neurons
% markings is the existing marking of temproal spikes. It can be empty.
% trace is the traces of all neurons. If there is not such an input, 
    % the program will calculate the trace according to video and masks.
% The users can adjust thred, DiamRatio, and ratio_ylim according to their needs.
    % thred controls the spike threshold. 
    % DiamRatio controls the size of the display region.
    % ratio_ylim controls the yaix limit of the trace plot.
    
    global i previous;
    global result resultString resultSpikes;
    global CallBackInterrupted;
    CallBackInterrupted = 0;
    global IsPlaying
    global timeline;
    timeline = [];
    global gui;
    global Trace;
%     Trace = 0;
%     global ARR; global countARR;
%     ARR = []; countARR = [];
    global currentNeuronArr;
    currentNeuronArr = [];
    global spikeCheck spikeArray spikeCount spikeTotal onSpike yesNoArray;
    spikeArray = [];
    spikeCheck = false;
    onSpike = false;
    spikeCount = 1;
    yesNoArray = [];
%     meanAc = 0;
    baselineAc = 0;
    sigma = 0;
    global baseline_method std_method;
    baseline_method = 'ksd';
%     std_method = 'median-based std'; 
    std_method = 'psd';
%     thred = 2; % for ABO using median-std
%     thred = 10; % for ABO use ksd-psd
    thred = 10; % for one photon
%     thred = 20; % for one photon
    global ratio_ylim;
    ratio_ylim=[-3,15]*thred/6;
    global data1;
    global mask;
%     global fileout;
    global FullVideoIsPlaying;
    addpath(genpath(pwd));
 
    global hasMarkings;
    global A;
    A = [];
    global AIndices;
    AIndices = [];
    count = 1;
    global COM
    global r_cum
 
    global DiamRatio;
    DiamRatio = 3;
    global pause_time; 
    pause_time = 0.1;
    global gap;
    gap = 6;
%     global old_Spikes;

    global color
    color=[  0    0.4470    0.7410
        0.8500    0.3250    0.0980
        0.9290    0.6940    0.1250
        0.4940    0.1840    0.5560
        0.4660    0.6740    0.1880
        0.3010    0.7450    0.9330
        0.6350    0.0780    0.1840];
    color = color([1,3,4,5,6],:);
    
    if nargin < 3 || isempty(markings)
        hasMarkings = false;
        resultSpikes=cell(size(Mask,3),1);
        %%disp('000');
    else
        hasMarkings = true;
        resultSpikes=markings;
        A = [];
        AIndices = [];
        totalCount = 1;
        for o = 1:length(markings)
            curr = markings{o};
            if ~(size(curr, 1) == 0)
                currCount = 0;
                for p = 1:size(curr, 1)
                    startTime = curr(p);
                    endTime = curr(p + size(curr, 1));
                    isNeuron = curr(p + 2*size(curr, 1));
                    if(isNeuron == 1)
                        A = [A startTime endTime];
                        currCount = currCount + 2;
                    end
                end
                AIndices = [AIndices o totalCount totalCount+currCount-1];
                totalCount = totalCount+currCount;
            end
        end
    end
%     resultSpikes=old_Spikes;
 
%     fileout = fopen('results/out.txt', 'w');
    
    if nargin<4
        Trace=0;
    else
        Trace=trace;
    end
 
    %% Get traces for each mask
    [x, y, T] = size(vid);
    n=size(Mask,3);
    if (Trace == 0)
        v = reshape(vid, x*y, T);
        guimask = Mask;
        guitrace = zeros(size(guimask, 3), T);
        for k = 1:size(guimask, 3)
            guitrace(k, :) = sum(v(reshape(guimask(:, :, k), [], 1) == 1, :), 1);
        end
        guitrace = guitrace ./ median(guitrace, 2) - 1;
     
        Trace = guitrace;
    end
    assignin('base', 'ans', Trace);
    %bin input video f    global gui data1; or faster visualization
    video_class=class(vid);
    scale = 1;
    %     FR = 6/scale;
    if scale>1
        vid = binVideo_temporal(vid, scale); % YB 2019/07/22
        Trace = double(binTraces_temporal(Trace, scale));
        if contains(video_class,'int')
            eval(['vid = ',video_class,'(vid);']);
        end
    end
 
    %% Get mean and std deviation for trace
%     meanAc = mean(Trace, 2); % YB 2019/08/22
%     sigma = std(Trace, 1, 2);
%     baselineAc = median(Trace, 2);
%     sigma = median(abs(Trace - baselineAc),2)/(sqrt(2)*erfinv(1/2)); 
    [baselineAc, sigma] = SNR_normalization(Trace,std_method,baseline_method);
 
    %% adjust contrast of frames
%     if strcmp(video_class,'single') || strcmp(video_class,'double')
%         vid=(vid-min(min(min(vid))))/(max(max(max(vid)))-min(min(min(vid))));
%     end

%     disp('Processing video for better visibility. May take several minutes.');
% %     for ii = 1:size(vid, 3)
% %         vid(:, :, ii) = imadjust(vid(:, :, ii), [], [], 0.5);
% %     end
%     vid(vid<0)=0;
%     video_max=prctile(vid(:),99);
%     vid(vid>video_max)=video_max;
%     video_adjust=sqrt(vid);
%     video_adjust_min=min(min(min(video_adjust)));
%     video_adjust_max=max(max(max(video_adjust)));
%     vid=(video_adjust-video_adjust_min)/(video_adjust_max-video_adjust_min);
%     clear video_adjust;
% %     if strcmp(video_class,'uint8')
% %         vid=uint8(vid*2^8-0.5);
% %     elseif strcmp(video_class,'uint16')
% %         vid=uint16(vid*2^16-0.5);
% %     end
%%% Video adjustment can be moved outside the GUI to prevent repeated calculation.
 
    [data1.d1, data1.d2, data1.T] = size(vid);
    data1.T = size(Trace, 2);
    data1.tracelimitx = [1, data1.T];
    data1.tracelimity = [floor(min(Trace(:))), ceil(max(Trace(:)))];
    data1.green = cat(3, zeros(data1.d1, data1.d2), ones(data1.d1, data1.d2), zeros(data1.d1, data1.d2));
 
    MAXImg = imadjust(max(vid, [], 3), [], [], 1.2);
    data1.maxImg = MAXImg;
 
    i = 1; %i = current neuron
 
    %% Calculate center of mass of all neurons
    r_cum=round(sqrt(mean(sum(sum(Mask))))*1.5);
    COM=zeros(n,2);
    for nn=1:n
        [xxs,yys]=find(Mask(:,:,nn)>0);
        COM(nn,:)=[mean(xxs),mean(yys)];
    end

    %-------------------------------------------------------------------------%
    createInterface();
    updateInterface();
    ResetSpikeArray();
    setListBox();
    
    %% createInterface - initialize GUI
    function createInterface()
     
        gui = struct();
        screensize = get(groot, 'ScreenSize');
        gui.Window = figure(...
          'Name', 'Select Neurons', ...
          'NumberTitle', 'off', ...
          'MenuBar', 'none', ...
          'Toolbar', 'none', ...
          'HandleVisibility', 'off', ...
          'Position', [screensize(3) / 9, screensize(4) / 9, screensize(3) * 7/9, screensize(4) * 7/9] ...
          );
     
        % Arrange the main interface
        mainLayout = uix.VBoxFlex(...
        'Parent', gui.Window, ...
          'Spacing', 3);
        upperLayout = uix.HBoxFlex(...
          'Parent', mainLayout, ...
          'Padding', 3);
        lowerLayout = uix.HBoxFlex(...
          'Parent', mainLayout, ...
          'Padding', 3);
     
        % Upper Layout Design
        gui.MaskPanel = uix.BoxPanel(...
        'Parent', upperLayout, ...
          'Padding', 3, ...
          'Title', 'Mask');
        gui.MaskAxes = axes('Parent', gui.MaskPanel);
     
        gui.VideoPanel = uix.BoxPanel(...
          'Parent', upperLayout, ...
          'Title', 'Video');
        gui.VideoAxes = axes(...
          'Parent', gui.VideoPanel, ...
          'ButtonDownFcn', @PlayVideo, ...
          'HitTest', 'on');
     
        gui.ListPanel = uix.VBoxFlex(...
          'Parent', upperLayout);
        gui.ListBox = uicontrol(...
          'Style', 'ListBox', ...
          'Parent', gui.ListPanel, ...
          'FontSize', 10, ...
          'String', {}, ...
          'CallBack', @MoveToNeuron);
        gui.ListBoxSpikes = uicontrol(...
          'Style', 'ListBox', ...
          'Parent', gui.ListPanel, ...
          'FontSize', 10, ...
          'String', {}, ...
          'CallBack', @MoveToSpike);
        gui.ListFB = uix.HBoxFlex(...
          'Parent', gui.ListPanel, ...
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
        set(gui.ListPanel, 'Heights', [- 2.5, - 2.5, - 1]);
     
        set(upperLayout, 'Widths', [- 2.33, - 1.67, - 1]);
     
        % Lower Layout Design
        gui.TracePanel = uix.BoxPanel(...
        'Parent', lowerLayout, ...
          'Title', 'Trace');
        gui.TraceAxes = axes('Parent', gui.TracePanel);
     
        gui.ControlPanel = uix.VBoxFlex(...
          'Parent', lowerLayout);
        gui.ControlPanel2 = uix.VBoxFlex(...
          'Parent', lowerLayout);
        FullVideoIsPlaying = false;
        gui.PlayButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel, ...
          'String', 'Play Video', ...
          'CallBack', @PlayVideo);
        gui.YesButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel, ...
          'String', 'Yes Active', ...
          'CallBack', @YesActive);
        gui.NoButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel, ...
          'String', 'No Active', ...
          'CallBack', @NoActive);
        gui.SaveTraceButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel, ...
          'String', 'Save Changes', ...
          'CallBack', @SaveTrace);
        set(lowerLayout, 'Widths', [- 2.5, - 1, - 1]);
        gui.SetSpikeButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel2, ...
          'String', 'Select Spike', ...
          'CallBack', @PushSpikeButton);
        gui.YesSpikeButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel2, ...
          'String', 'Yes Spike', ...
          'CallBack', @YesSpike);
        gui.NoSpikeButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel2, ...
          'String', 'No Spike', ...
          'CallBack', @NoSpike);
        gui.ReplaySpikeButton = uicontrol(...
          'Style', 'PushButton', ...
          'Parent', gui.ControlPanel2, ...
          'String', 'Replay Spike Video', ...
          'CallBack', @ReplaySpike);
     
    end % createInterface
 
    %-------------------------------------------------------------------------%
 
    %% updateInterface - update GUI for new trace
    function updateInterface()
        % Update the Trace
        cla(gui.TraceAxes);
        data1.trace = Trace(i, :);
        plot(gui.TraceAxes, data1.trace);
        hold(gui.TraceAxes, 'on');
        plot(gui.TraceAxes, ones(size(data1.trace))*(baselineAc(i)+sigma(i)*thred),':','Color',[0,0.75,0]);
     
        data1.tracelimity = baselineAc(i)+sigma(i)*ratio_ylim;
%         data1.tracelimity = [min(data1.trace), max(data1.trace);
        [data1.pk, data1.lk] = findpeaks(data1.trace, 1:data1.T, 'MinPeakDistance', 5, 'MinPeakProminence', sigma(i)*thred/3, 'MinPeakHeight', baselineAc(i)+sigma(i)*thred/2); %
        if ~ isempty(data1.lk)
            plot(gui.TraceAxes, data1.lk, min(data1.pk,data1.tracelimity(2)), 'v','Color',[1,0.5,0]);
            mm = normalizeValues(mean(vid(:, :, data1.lk), 3));
            mm = imadjust(mm, stretchlim(mm, 0.02), []);
            data1.maxImg = imadjust(mm, [], [], .5);
        else
            data1.maxImg = MAXImg;
        end
        xlabel(gui.TraceAxes, 'Time(frame)');
        ylabel(gui.TraceAxes, 'SNR'); %deltaf/f*100
%           'Xlim', data1.tracelimitx, 'Ylim', [min(data1.trace), max(data1.trace)], ...
        set(gui.TraceAxes, ...
          'Xlim', data1.tracelimitx, 'Ylim', data1.tracelimity, ...
          'Units', 'normalized', 'Position', [0.1, 0.15, 0.8, 0.7]);

      % Update Video
        cla(gui.VideoAxes);
        mask = Mask(:, :, i);
        %             data1.mask = mask;
        data1.mask = mat2gray(mask);
        bw = mask>0;
%         bw(bw > 0) = 1;
        temp = regionprops(bw, data1.mask, 'WeightedCentroid','Area');
        if isempty(temp)
            temp1 = reshape((mean(data1.mask, 1) > 0), 1, []);
            size(temp1)
            [~, temp2] = find(temp1, 1);
            [~, temp3] = find(temp1, 1, 'last');
         
            data1.center(1) = mean([temp2, temp3]);
            temp1 = reshape((mean(mask, 2) > 0), 1, []);
            size(temp1)
            [~, temp2] = find(temp1, 1);
            [~, temp3] = find(temp1, 1, 'last');
         
            data1.center(2) = mean([temp2, temp3]);
        elseif length(temp)==1
            data1.center = round(temp.WeightedCentroid);
        else
            maxArea = temp(1).Area;
            data1.center = round(temp(1).WeightedCentroid);
            for ind = 2:length(temp)
                if temp(ind).Area > maxArea
                    data1.center = round(temp(ind).WeightedCentroid);
                    maxArea = temp(ind).Area;
                end
            end
        end
        measurements = regionprops(Mask(:, :, i), 'EquivDiameter');
        avgDiam = [measurements.EquivDiameter];
        avgDiam = round(mean(avgDiam) * DiamRatio);
     
        data1.boxy1 = max(data1.center(1) - avgDiam, 1);
        data1.boxy2 = min(data1.center(1) + avgDiam, data1.d2);
        data1.boxx1 = max(data1.center(2) - avgDiam, 1);
        data1.boxx2 = min(data1.center(2) + avgDiam, data1.d1);
        hold(gui.MaskAxes, 'on');
     
        if ~ isempty(data1.lk)
            [~,lk] = max(data1.pk); 
            imagesc(gui.VideoAxes, vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, max(data1.lk(lk), 1)));
        else
            imagesc(gui.VideoAxes, vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, 1));
        end
     
        hold(gui.VideoAxes, 'on');
     
        data1.videomask = data1.mask(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2);
        set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
          'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)]);
        gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
     
        % Update the Mask
        data1.masky1 = data1.boxy1;
        data1.masky2 = data1.boxy2;
        data1.maskx1 = data1.boxx1;
        data1.maskx2 = data1.boxx2;
     
        mask = data1.mask(data1.maskx1:data1.maskx2, data1.masky1:data1.masky2);
        axes(gui.MaskAxes);
        mm = data1.maxImg(data1.maskx1:data1.maskx2, data1.masky1:data1.masky2);
        imshow(mm, 'Parent', gui.MaskAxes, 'DisplayRange', []);
        hold(gui.MaskAxes, 'on');
     
        colormap(gui.MaskAxes, gray);
     
        gui.rectangle2 = rectangle(gui.MaskAxes, 'Position', [data1.center(1) - data1.masky1 - 6, data1.center(2) - data1.maskx1 - 6, 13, 13], 'EdgeColor', 'yellow');
        colormap(gui.VideoAxes, gray);
        set(gui.MaskAxes, ...
          'DataAspectRatio', [1, 1, 1], ...
          'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)]);
        hold(gui.MaskAxes, 'on');
        hold(gui.VideoAxes, 'on');
        IsPlaying = 0;
        CallBackInterrupted = 0;
        
        r = sqrt(sum((COM(i,:)-COM).^2,2)); 
        neighbors = intersect(find(r > 0),find(r < r_cum))'; 
        for j=neighbors
            contour(gui.MaskAxes, Mask(data1.maskx1:data1.maskx2, data1.masky1:data1.masky2,j), 1, 'LineColor', color(mod(find(j==neighbors),5)+1,:), 'linewidth', 1); % 'g'
        end            
        contour(gui.MaskAxes, mask, 1, 'LineColor', 'r', 'linewidth', 1);
    end
 
    %% SaveTrace - saves labeling output file and closes GUI window
    function SaveTrace(~, ~)
        resultSpikes{i} = currentNeuronArr;
        if ~isempty(currentNeuronArr)
            if any(currentNeuronArr(:,3))
                result(i) = 1;
                resultString{i} = sprintf('%d Yes', i);
            elseif result(i) == 2
                result(i) = 0;
                resultString{i} = sprintf('%d No', i);
            end
        end
        %         save('FinalTrace.mat', Trace);
     
%         output = {};
%         resultSpikes=old_Spikes;
     
%         count = 1;
%         count1 = 2;
%         tmp = size(countARR) / 2;
%         disp(countARR);
        disp(resultSpikes);
%         for b = 1:tmp(2)
%             currIndex = countARR(2 * b - 1);
% %             if(length(output) < currIndex)
% %                 for c = length(output)+1:currIndex
% %                     output{end+1,1} = [];
% %                 end
% %             end
% %             out(end + 1, 1) = {char(int2str(countARR(2 * b - 1)))};
%             for c = count:(count + countARR(2 * b) - 1)
% %                 resultSpikes{currIndex,1}(end+1, 1:3) = [ARR(3 * c - 2) ARR(3 * c - 1) ARR(3 * c)];
%                 resultSpikes{currIndex,1}(end+1, 1:3) = ARR(c,:);
%                 count1 = count1 + 1;
%             end
%             disp(resultSpikes{currIndex,1});
%             count1 = 2;
%             count = count + countARR(2 * b);
%          
%         end
        
%         num_output=length(output)-sum(cellfun(@isempty, output));
        fileName=fullfile('Results',['transients_',replace(char(datetime),':','-')]);
%         while exist([fileName,'.mat'],'file')
%             fileName=[fileName,'+'];
%         end
        save([fileName,'.mat'], 'resultSpikes');
        close(gui.Window);
    end
 
    %% ResetSpikeArray - Reset spike variables, calculate mean and standard
    %deviation for trace, loop through trace to check for spikes of
    %activity
    function ResetSpikeArray(~, ~)
        currentNeuronArr = [];
        spikeArray = [];
        spikeCheck = false;
        onSpike = false;
        spikeCount = 1;
        yesNoArray = [];
%         meanAc = mean(Trace, 2); % YB 2019/08/22
%         sigma = std(Trace, 1, 2);
%         baselineAc = median(Trace, 2);
%         sigma = median(abs(Trace - baselineAc),2)/(sqrt(2)*erfinv(1/2)); 
        [baselineAc, sigma] = SNR_normalization(Trace,std_method,baseline_method);
        %           for n = 1:size(Mask, 3)
        foundPeak = false;
        hasNeuronMarkings = false;
        if (hasMarkings)
            for z = 0:(size(AIndices, 2) / 3) - 1
                if (isequal(AIndices(3 * z + 1), i))
                    hasNeuronMarkings = true;
                    for k = AIndices(3 * z + 2):AIndices(3 * z + 3)
                        spikeArray = [spikeArray A(k)];
                    end
                end
            end
        end
        if ~ hasNeuronMarkings
            for j = 1:size(vid, 3)
                %             if (Trace(i, j) - meanAc) / sigma > 5
                if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred
                    %                    gui.rectangle = rectangle(gui.VideoAxes,'Position',[data1.center(1)-data1.boxy1-6,data1.center(2)-data1.boxx1-6,13,13],'EdgeColor','red');
                    if ~ onSpike
                        spikeArray = [spikeArray, j];
                        onSpike = true;
                    end
                    if (j==T || Trace(i, j + 1) < Trace(i, j)) && ~ foundPeak
                        plot(gui.TraceAxes, j, min(Trace(i, j),data1.tracelimity(2)), '-s', 'MarkerSize', 10, 'MarkerEdgeColor', 'red');
                        foundPeak = true;
                    end
                else
                    if onSpike
                        spikeArray = [spikeArray, j-1];
                        onSpike = false;
                    end
                    foundPeak = false;
                end
            end
            if onSpike
                spikeArray = [spikeArray, j];
%                 onSpike = false;
            end
            % Split transients with multiple prominent peaks. 
            % Eliminate transients without any prominent peak. 
            starts = spikeArray(1:2:end);
            ends = spikeArray(2:2:end);
            num_active = length(starts);
            [~, locs] = findpeaks(Trace(i,:), 1:data1.T, 'MinPeakProminence', sigma(i)*thred/3, 'MinPeakHeight', baselineAc(i)+sigma(i)*thred); % 'MinPeakDistance', 5, 
            if Trace(i,1)>baselineAc(i)+sigma(i)*thred
                locs = [1, locs];
            end
            if Trace(i,end)>baselineAc(i)+sigma(i)*thred
                locs = [locs, data1.T];
            end
            spikes_peaks=false(1,data1.T);
            spikes_peaks(locs)=true;
%             spikes_peaks([1,end])=Trace(i,[1,end])>baselineAc(i)+sigma(i)*thred;
            list_temp_spikes_split = cell(num_active,1);
            for ii=1:num_active
                num_spikes = sum(spikes_peaks(starts(ii):ends(ii)));
                if num_spikes==0
                    temp_spikes_split = zeros(0,2);
                elseif num_spikes==1
                    temp_spikes_split = [starts(ii),ends(ii)];
                elseif num_spikes>1
                    list_locs = locs((locs >= starts(ii)) & (locs <= ends(ii)));
                    temp_spikes_split = zeros(num_spikes,2);
                    temp_spikes_split(1,1) = starts(ii);
                    temp_spikes_split(num_spikes,2) = ends(ii);
                    for jj = 1:num_spikes-1
                        [~,ind] = min(Trace(i,list_locs(jj):list_locs(jj+1)));
                        valley = ind+list_locs(jj)-1;
                        temp_spikes_split(jj,2) = valley-1;
                        temp_spikes_split(jj+1,1) = valley+1;
                    end
                end
                list_temp_spikes_split{ii} = temp_spikes_split;
            end
            spikeArray_2 = cell2mat(list_temp_spikes_split);
            spikeArray = reshape(spikeArray_2',1,[]);
        end
        for j = 1:size(spikeArray, 2) / 2
            tmpmax = -inf;
            tmpmaxindex = 0;
            for z = spikeArray(2 * j - 1):spikeArray(2 * j)
                if (Trace(i, z) > tmpmax)
                    tmpmax = Trace(i, z);
                    tmpmaxindex = z;
                end
            end
            if tmpmaxindex > 0
                plot(gui.TraceAxes, tmpmaxindex, min(Trace(i, tmpmaxindex),data1.tracelimity(2)), '-s', 'MarkerSize', 10, ...
                  'MarkerEdgeColor', 'red');
            end
        end
     
        %           end
        %           for n = 1:size(Mask, 3)
     
        %           end
        spikeTotal = size(spikeArray, 2) / 2;
        tmp = [];
        for g = 1:spikeTotal
            tmp = [tmp g];
        end
        set(gui.ListBoxSpikes, 'String', tmp);
        if (spikeTotal > 0)
            set(gui.ListBoxSpikes, 'Value', spikeCount);
        end
    end
 
    %% PlayVideo - play full video for selected neuron
    function PlayVideo(~, ~)
        if IsPlaying == 1
            IsPlaying = 0;
            return;
        else
            IsPlaying = 1;
            %             if ishandle(timeline); delete(timeline); end
            if ~ isempty(data1.lk)
                playduration = [max(data1.lk(1) - 60, 1), min(data1.lk(end) + 60, size(Trace, 2))];
            else
                playduration = [1, data1.T];
            end
         
            hold(gui.VideoAxes, 'on');
            %gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
            %gui.smallgreen = image(gui.VideoAxes, data1.smallgreen, 'Alphadata', data1.videomask);
         
            %pause(0.4);
            %delete(gui.smallgreen);
            pause(0.4);
            hold(gui.VideoAxes, 'off');
            currentylim = get(gui.TraceAxes, 'Ylim');
            temp = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, :);
            cmin = min(temp(:));
            %cmax = max([0.6*max(temp(:)) cmin]);
            cmax = max([0.9 * max(temp(:)), cmin]);
            for j = playduration(1):playduration(2)
                if (IsPlaying == 0)
                    break;
                end
                imgShow = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, j);
                imagesc(gui.VideoAxes, imgShow);
                set(gui.VideoAxes, 'clim', [cmin, cmax]);
                hold(gui.VideoAxes, 'on');
                colormap(gui.VideoAxes, gray);
                %contour(gui.VideoAxes, mask, 1, 'LineColor', 'black', 'linewidth', 1);
             
                %             if (Trace(i, j) - meanAc) / sigma > 5
                % if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred  % YB 2019/08/22
                tmpbool = false;
                for a = 1:size(spikeArray, 2) / 2
                    if Trace(i, j) < spikeArray(2 * a) && Trace(i, j) > spikeArray(2 * a - 1)
                        tmpbool = true;
                    end
                end
                if tmpbool
                    gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'red');
                else
                    gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                end
                set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                  'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                  'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                  'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
             
                timeline = plot(gui.TraceAxes, [j, j], currentylim, '-', 'Color', 'red');
                pause(pause_time);
                delete(timeline);
             
                if CallBackInterrupted
                    CallBackInterrupted = 0;
                    IsPlaying = 0;
                    return;
                end
             
                %To prevent freeze in video
                hold(gui.VideoAxes, 'off');
            end
        end
    end
 
    %% PushSpikeButton - start spike labeling, play video for 1st spike
    function PushSpikeButton(~, ~)
        delete(timeline);
        spikeCheck = true;
     
        if (spikeTotal > 0)
            %fprintf(fileout, strcat(int2str(spikeArray(1)), ' ', int2str(spikeArray(2)), ' '));
            foundPeak = false;
            currentylim = get(gui.TraceAxes, 'Ylim');
            set(gui.VideoPanel, 'Title', strcat('Video - Spike at frame #', int2str(spikeArray(2 * spikeCount - 1))));
            set(gui.ListBoxSpikes, 'Value', spikeCount);
            for j = max(spikeArray(1) - gap, 1):min(spikeArray(2) + gap, size(vid, 3))
                IsPlaying = 1;
                imgShow = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, j);
                imagesc(gui.VideoAxes, imgShow);
                hold(gui.VideoAxes, 'on');
                %                 if (Trace(i, j) - meanAc) / sigma > 5
%                 if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred % YB 2019/08/22
                if j >= spikeArray(1) && j<=spikeArray(2)
                    gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'red');
                 
                else
                    gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                    foundPeak = false;
                end
                set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                  'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                  'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                  'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
                colormap(gui.VideoAxes, gray);
                timeline = plot(gui.TraceAxes, [j, j], currentylim, '-', 'Color', 'red');
                pause(pause_time);
                delete(timeline);
             
                if CallBackInterrupted
                    CallBackInterrupted = 0;
                    IsPlaying = 0;
                    return;
                end
             
            end
            imgMean=mean(vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, spikeArray(2 * spikeCount - 1):spikeArray(2 * spikeCount)),3);
            imagesc(gui.VideoAxes, imgMean);
            hold(gui.VideoAxes, 'on');
            set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
              'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
              'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
              'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
            colormap(gui.VideoAxes, gray);
            gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
            timeline = plot(gui.TraceAxes, [spikeArray(1), spikeArray(2)], currentylim, '-', 'Color', 'red');
        end
        IsPlaying = 0;
    end
 
    %% YesSpike - Check if user is looking for spikes, print to output file,
    %and play video for next spike
    function YesSpike(~, ~)
%         if IsPlaying
%             CallBackInterrupted = 1;
%             pause(0.1);
%         end
        delete(timeline);
        if spikeCheck
            currentSpikeArr = [spikeArray(2 * spikeCount - 1) spikeArray(2 * spikeCount) 1];
            currentNeuronArr = [currentNeuronArr; currentSpikeArr];
            spikeCount = spikeCount + 1;
         
%             ARR = [ARR, currentSpikeArr];
%             if (size(countARR, 2) == 0)
%                 countARR = [countARR i 1];
%             else
%                 if (countARR(size(countARR, 2) - 1) == i)
%                     countARR(size(countARR, 2)) = countARR(size(countARR, 2)) + 1;
%                 else
%                     countARR = [countARR i 1];
%                 end
%             end
         
            if spikeCount > spikeTotal
                spikeCheck = false;
                spikeCount = 1;
%                 currentNeuronArr = [];
             
            else
                set(gui.ListBoxSpikes, 'Value', spikeCount);
                currentylim = get(gui.TraceAxes, 'Ylim');
                set(gui.VideoPanel, 'Title', strcat('Video - Spike at frame #', int2str(spikeArray(2 * spikeCount - 1))));
             
                for j = max(spikeArray(2 * spikeCount - 1) - gap, 1):min(spikeArray(2 * spikeCount) + gap, size(vid, 3))
                    IsPlaying = 1;
                    imgShow = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, j);
                    imagesc(gui.VideoAxes, imgShow);
                    hold(gui.VideoAxes, 'on');
%                     if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred % YB 2019/08/22
                    if j >= spikeArray(2 * spikeCount - 1) && j<=spikeArray(2 * spikeCount)
                        gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'red');
                    else
                        gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                    end
                    set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                      'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                      'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                      'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
                    colormap(gui.VideoAxes, gray);
                    timeline = plot(gui.TraceAxes, [j, j], currentylim, '-', 'Color', 'red');
                    pause(pause_time);
                    delete(timeline);
                 
                    if CallBackInterrupted
                        CallBackInterrupted = 0;
                        IsPlaying = 0;
                        return;
                    end
                end
                 
                imgMean=mean(vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, spikeArray(2 * spikeCount - 1):spikeArray(2 * spikeCount)),3);
                imagesc(gui.VideoAxes, imgMean);
                hold(gui.VideoAxes, 'on');
                set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                  'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                  'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                  'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
                colormap(gui.VideoAxes, gray);
                gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                %To prevent freeze in video
                hold(gui.VideoAxes, 'off');
                timeline = plot(gui.TraceAxes, [spikeArray(2 * spikeCount - 1), spikeArray(2 * spikeCount)], currentylim, '-', 'Color', 'red');
            end
        end
        IsPlaying = 0;
    end
 
    %% NoSpike - Check if user is looking for spikes, print to output file,
    %and play video for next spike
    function NoSpike(~, ~)
%         if IsPlaying
%             CallBackInterrupted = 1;
%             pause(0.1);
%         end
        delete(timeline);
        if spikeCheck
            currentSpikeArr = [spikeArray(2 * spikeCount - 1) spikeArray(2 * spikeCount) 0];
            currentNeuronArr = [currentNeuronArr; currentSpikeArr];
            spikeCount = spikeCount + 1;
%             ARR = [ARR, currentSpikeArr];
%             if (size(countARR, 2) == 0)
%                 countARR = [countARR i 1];
%             else
%                 if (countARR(size(countARR, 2) - 1) == i)
%                     countARR(size(countARR, 2)) = countARR(size(countARR, 2)) + 1;
%                 else
%                     countARR = [countARR i 1];
%                 end
%             end
         
            if spikeCount > spikeTotal
                spikeCheck = false;
                spikeCount = 1;
%                 currentNeuronArr = [];
            else
                set(gui.ListBoxSpikes, 'Value', spikeCount);
                currentylim = get(gui.TraceAxes, 'Ylim');
                set(gui.VideoPanel, 'Title', strcat('Video - Spike at frame #', int2str(spikeArray(2 * spikeCount - 1))));
             
                for j = max(spikeArray(2 * spikeCount - 1) - gap, 1):min(spikeArray(2 * spikeCount) + gap, size(vid, 3))
                    IsPlaying = 1;
                    imgShow = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, j);
                    imagesc(gui.VideoAxes, imgShow);
                    hold(gui.VideoAxes, 'on');
%                     if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred % YB 2019/08/22
                    if j >= spikeArray(2 * spikeCount - 1) && j<=spikeArray(2 * spikeCount)
                        gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'red');
                    else
                        gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                    end
                    set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                      'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                      'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                      'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
                    colormap(gui.VideoAxes, gray);
                    timeline = plot(gui.TraceAxes, [j, j], currentylim, '-', 'Color', 'red');
                    pause(pause_time);
                    delete(timeline);
                 
                    if CallBackInterrupted
                        CallBackInterrupted = 0;
                        IsPlaying = 0;
                        return;
                    end
                end
                 
                imgMean=mean(vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, spikeArray(2 * spikeCount - 1):spikeArray(2 * spikeCount)),3);
                imagesc(gui.VideoAxes, imgMean);
                hold(gui.VideoAxes, 'on');
                set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                  'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                  'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                  'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
                colormap(gui.VideoAxes, gray);
                gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                %To prevent freeze in video
                hold(gui.VideoAxes, 'off');
                timeline = plot(gui.TraceAxes, [spikeArray(2 * spikeCount - 1), spikeArray(2 * spikeCount)], currentylim, '-', 'Color', 'red');
            end
        end
        IsPlaying = 0;
    end

    function MoveToSpike(src, ~)
        delete(timeline);
        persistent chk
        if isempty(chk)
            chk = 1;
            pause(0.5); %Add a delay to distinguish single click from a double click
            if chk == 1
                chk = [];
            end
        else
            chk = [];
            % function starts here
            spikeCheck = true;
            spikeCount = get(src, 'Value');
            currentylim = get(gui.TraceAxes, 'Ylim');
            set(gui.VideoPanel, 'Title', strcat('Video - Spike at frame #', int2str(spikeArray(2 * spikeCount - 1))));
            for j = max(spikeArray(2 * spikeCount - 1) - gap, 1):min(spikeArray(2 * spikeCount) + gap, size(vid, 3))
                IsPlaying = 1;
                imgShow = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, j);
                imagesc(gui.VideoAxes, imgShow);
                hold(gui.VideoAxes, 'on');
%                 if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred % YB 2019/08/22
                if j >= spikeArray(2 * spikeCount - 1) && j<=spikeArray(2 * spikeCount)
                    gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'red');
                else
                    gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
                end
                set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
                  'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
                  'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
                  'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
                colormap(gui.VideoAxes, gray);
             
                timeline = plot(gui.TraceAxes, [j j], currentylim, '-', 'Color', 'red');
                pause(pause_time);
                delete(timeline);
             
                if CallBackInterrupted
                    CallBackInterrupted = 0;
                    IsPlaying = 0;
                    return;
                end
             
                %To prevent freeze in video
                hold(gui.VideoAxes, 'off');
            end
            imgMean=mean(vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, spikeArray(2 * spikeCount - 1):spikeArray(2 * spikeCount)),3);
            imagesc(gui.VideoAxes, imgMean);
            hold(gui.VideoAxes, 'on');
            set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
              'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
              'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
              'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
            colormap(gui.VideoAxes, gray);
            gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
            set(gui.ListBoxSpikes, 'Value', spikeCount);
            timeline = plot(gui.TraceAxes, [spikeArray(2 * spikeCount - 1), spikeArray(2 * spikeCount)], currentylim, '-', 'Color', 'red');
         
        end
    end

    %% YesActive - currently not used
    function YesActive(~, ~)
        %fprintf(fileout, strcat(int2str(i), ' 1\n'));
        result(i) = 1;
        resultString{i} = sprintf('%d Yes', i);
        previous = i;
        if ~ (i == size(Mask, 3))
            i = i + 1;
        end
%         i = previous +1;
%         i = find(result == 2, 1);

        if IsPlaying
            CallBackInterrupted = 1;
        end

        if ~isempty(i)
            updateInterface();
            ResetSpikeArray();
            set(gui.ListBox, 'Value', i);
            set(gui.ListBox, 'String', resultString);
    %         else
%             close(gui.Window);
        end
    end
 
    %% NoActive - currently not used
    function NoActive(~, ~)
        %fprintf(fileout, strcat(int2str(i), ' 0\n'));
        result(i) = 0;
        resultString{i} = sprintf('%d No', i);
        previous = i;
        if ~ (i == size(Mask, 3))
            i = i + 1;
        end
%         i = previous +1;
%         i = find(result == 2, 1);

        if IsPlaying
            CallBackInterrupted = 1;
        end

        if ~isempty(i)
            updateInterface();
            ResetSpikeArray();
            set(gui.ListBox, 'Value', i);
            set(gui.ListBox, 'String', resultString);
    %         else
%             close(gui.Window);
        end
    end
 
    %% Forward - move current neuron forward 1 and update interface
    function Forward(~, ~)
        resultSpikes{i} = currentNeuronArr;
        if ~isempty(currentNeuronArr)
            if any(currentNeuronArr(:,3))
                result(i) = 1;
                resultString{i} = sprintf('%d Yes', i);
            elseif result(i) == 2
                result(i) = 0;
                resultString{i} = sprintf('%d No', i);
            end
        end

        if ~ (i == size(Mask, 3))
            i = i + 1;
            updateInterface();
            ResetSpikeArray();
            set(gui.ListBox, 'Value', i);
        end
        set(gui.ListBox, 'String', resultString);
     
    end
 
    %% Backward - move current neuron backward 1 and update interface
    function Backward(~, ~)
        resultSpikes{i} = currentNeuronArr;
        if ~isempty(currentNeuronArr)
            if any(currentNeuronArr(:,3))
                result(i) = 1;
                resultString{i} = sprintf('%d Yes', i);
            elseif result(i) == 2
                result(i) = 0;
                resultString{i} = sprintf('%d No', i);
            end
        end

        if ~ (i == 1)
            i = i - 1;
            updateInterface();
            ResetSpikeArray();
            set(gui.ListBox, 'Value', i);
        end
        set(gui.ListBox, 'String', resultString);
    end
 
    %% setListBox - set box for list of neurons using mask array size
    function setListBox()
%         arr = [];
%         for c = 1:size(Mask, 3)
%             arr = [arr, c];
%         end
        set(gui.ListBox, 'String', resultString); % arr
     
    end

    function MoveToNeuron(src, ~)
        resultSpikes{i} = currentNeuronArr;
        if ~isempty(currentNeuronArr)
            if any(currentNeuronArr(:,3))
                result(i) = 1;
                resultString{i} = sprintf('%d Yes', i);
            elseif result(i) == 2
                result(i) = 0;
                resultString{i} = sprintf('%d No', i);
            end
        end
        
        persistent chk
        if isempty(chk)
            chk = 1;
            pause(0.5); %Add a delay to distinguish single click from a double click
            if chk == 1
                chk = [];
            end
        else
            chk = [];
            i = get(src, 'Value');
            updateInterface();
            ResetSpikeArray();
            set(gui.ListBox, 'Value', i);
            set(gui.ListBox, 'String', resultString);
         
        end
        
    end
    function ReplaySpike(~, ~)
        delete(timeline);
        currentylim = get(gui.TraceAxes, 'Ylim');
        set(gui.VideoPanel, 'Title', strcat('Video - Spike at frame #', int2str(spikeArray(2 * spikeCount - 1))));
        for j = max(spikeArray(2 * spikeCount - 1) - gap, 1):min(spikeArray(2 * spikeCount) + gap, size(vid, 3))
            IsPlaying = 1;
            imgShow = vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, j);
            imagesc(gui.VideoAxes, imgShow);
            hold(gui.VideoAxes, 'on');
%             if (Trace(i, j) - baselineAc(i)) / sigma(i) > thred
            if j >= spikeArray(2 * spikeCount - 1) && j<=spikeArray(2 * spikeCount)
                gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'red');
            else
                gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
            end
            set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
              'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
              'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
              'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
            colormap(gui.VideoAxes, gray);
            timeline = plot(gui.TraceAxes, [j, j], currentylim, '-', 'Color', 'red');
            pause(pause_time);
            delete(timeline);
         
            if CallBackInterrupted
                CallBackInterrupted = 0;
                IsPlaying = 0;
                return;
            end
         
            %To prevent freeze in video
            hold(gui.VideoAxes, 'off');
        end
        imgMean=mean(vid(data1.boxx1:data1.boxx2, data1.boxy1:data1.boxy2, spikeArray(2 * spikeCount - 1):spikeArray(2 * spikeCount)),3);
        imagesc(gui.VideoAxes, imgMean);
        hold(gui.VideoAxes, 'on');
        set(gui.VideoAxes, 'DataAspectRatio', [1, 1, 1], ...
          'Xlim', [1, size(data1.videomask, 2)], 'Ylim', [1, size(data1.videomask, 1)], ...
          'XTick', 1:30:size(data1.videomask, 2), 'YTick', 1:30:size(data1.videomask, 1), ...
          'XTickLabel', data1.boxy1:30:data1.boxy2, 'YTickLabel', data1.boxx1:30:data1.boxx2);
        colormap(gui.VideoAxes, gray);
        gui.rectangle = rectangle(gui.VideoAxes, 'Position', [data1.center(1) - data1.boxy1 - 6, data1.center(2) - data1.boxx1 - 6, 13, 13], 'EdgeColor', 'yellow');
        timeline = plot(gui.TraceAxes, [spikeArray(2 * spikeCount - 1), spikeArray(2 * spikeCount)], currentylim, '-', 'Color', 'red');
        IsPlaying = 0;
    end
end