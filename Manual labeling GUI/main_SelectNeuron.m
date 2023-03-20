%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Proceedings of the National Academy of Sciences (PNAS), 2019.
%
% Released under a GPL v2 license.
%
%% USed to run the marking software

function main_SelectNeuron(vid,guimask,DirSave,name,video_SNR)
    [x,y,~] = size(vid);
    if isempty(guimask)
        guimask = zeros(x,y,0);
    end
    %% Manually add missing neurons
    saveName = [DirSave,filesep,'Added_',name,'.mat'];
if exist('video_SNR','var')
    SNRvideo = AddROI(vid,saveName, guimask,video_SNR);
else
    SNRvideo = AddROI(vid,saveName, guimask);
end
    fig = gcf;
    waitfor(fig);

    %%
    [x,y,T] = size(SNRvideo);
%     vid = vid - min(vid,[],3);
    v = reshape(SNRvideo,x*y,T);
%     % If there is no initial mask, starst from scratch
%     if isempty(guimask)
%         guimask = zeros(x,y);
%         guitrace = zeros(1,T);
%     else
%            %% Get traces for each mask
%         guitrace = zeros(size(guimask,3),T);        
%         for k = 1:size(guimask,3)
%             guitrace(k,:) = sum(v(reshape(guimask(:,:,k),[],1)==1,:),1);
%         end
%     end
    
    %% Load resulting masks     
%     numMasksOrig =size(guimask,3);
    load(saveName,'FinalMasks');
    guimask = FinalMasks;
    clear FinalMasks;
    m = reshape(guimask,[],size(guimask,3));
    guimask(:,:,sum(m,1)==0) = [];
    m(:,sum(m,1)==0) = [];
    ncells = size(guimask,3);
    
    %% Calculate traces from the masks
    guitrace = zeros(ncells,T);
    for k = 1:ncells
%         guitrace(k,:) = sum(v(reshape(guimask(:,:,k),[],1)==1,:),1);
        guitrace(k,:) = sum(v(m(:,k)==1,:),1);
    end

%     if size(guimask,3)>numMasksOrig
%         for k = numMasksOrig+1:size(guimask,3)
%             guitrace(k,:) = sum(v(reshape(guimask(:,:,k),[],1)==1,:),1);
%         end
%     end

%     %% Check for all zero masks
%     guitrace(sum(m,1)==0,:) = [];
    
    %% rough estimate of SNR or dff
%     guitrace = guitrace./median(guitrace,2) - 1; % df/f
    guitrace = (guitrace - median(guitrace,2))./std(guitrace,1,2); % df/f
    clear v
    %% Set global variables
%     ncells = size(guimask,3);
    global gui; global data1;
    global result;       
    global resultString; 
%     global resultSpTimes; 
    global resultSpikes; 
    if isempty(result)
        result = ones(1,ncells)*2;
    end
    if isempty(resultString)
%         for j = 1:size(guimask,3) 
%             resultString{j} = num2str(j); 
%         end
        resultString = arrayfun(@num2str, 1:ncells,'UniformOutput',false);
    end
%     resultSpTimes = cell(ncells,1);
    if isempty(resultSpikes)
        resultSpikes = cell(ncells,1);
    end
    
    n_current = length(result);
    if n_current < ncells
        n_add = ncells - n_current;
        result = [result, ones(1,n_add)*2];
        resultString = [resultString,arrayfun(@num2str, ...
            (n_current+1):ncells,'UniformOutput',false)];
        resultSpikes = [resultSpikes; cell(n_add,1)];
    end

    %% GUI main
    saveName = [DirSave,filesep,'FinalMasks_',name,'.mat'];
%     SelectNeuron(vid,guimask,guitrace)
    NeuronFilter(SNRvideo,guimask,resultSpikes,guitrace)
    waitfor(gui.Window)
    
%     for i = 1:ncells
%         Spikes = resultSpikes{i};
%         if ~isempty(Spikes)
%             if any(Spikes(:,3))
%                 result(i) = 1;
%                 resultString{i} = sprintf('%d Yes', i);
%             elseif result(i) == 2
%                 result(i) = 0;
%                 resultString{i} = sprintf('%d No', i);
%             end
%         end
%     end

    FinalMasks = guimask(:,:,result==1);
    FinalTimes = resultSpikes(result==1);
    list_text = {'',' Yes',' No'};
    resultString = arrayfun(@(x) [num2str(x),list_text{3-result(x)}],1:ncells,'UniformOutput',false);

    save(saveName,'FinalMasks','FinalTimes');

    saveName = [DirSave,filesep,'ManualIDs_traces_',name,'.mat'];
    save(saveName,'guimask','guitrace','result','resultString','resultSpikes')

    clear gui finalMask Y guimask guitrace result resultSpikes

end
