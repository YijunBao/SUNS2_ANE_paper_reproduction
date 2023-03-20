%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Submitted to PNAS.
%
% Released under a GPL v2 license.

function [Label,num_blocks] = prepareTemporalMask(DirData,DirMask,DirSave,name,tau,fs,lam,Pd, block_size)
% create temporal labeling of neurons

opt.tau = tau;
opt.fs = fs;
stat.thresh = norminv(Pd)-norminv(lam/(fs-lam)*(1-Pd));
stat.Pd = Pd;

%% If data is already processed, display message and don't do anything
if exist([DirSave,filesep,'TemporalMask_',name,'.nii.gz']) || exist([DirSave,filesep,'TemporalMask_',name,'_1.nii.gz'])%
    disp('Temporal label already exists.')
else
    f = dir([DirMask,filesep,'*_',name(1:3),'.mat']); %
    if ~isempty(f)
        MaskName = [f.folder,filesep,f.name];        
    else
        error('Mask not found.')
    end    
    
    matObj = matfile(MaskName);
    details = whos(matObj);
    
    if ~contains([details.name],'FinalMasks')
        error('Neuron Masks not found in Mask file. It should be named FinalMasks.')
    end
    
    load(MaskName,'FinalMasks');
    
    if ~contains([details.name],'FinalTimes')
        FinalTimes = [];
    else
        load(MaskName,'FinalTimes');
    end
    
    data_name = [DirData,filesep,sprintf('%05.2f',str2num(name(1:3))/100),name(4:end),'.h5']; %
%     data_name = [DirData,filesep,name,'.h5']; %.gz 
    if ~exist(data_name)
        error('Downsampled, Cropped Data not found.')
    else
%         vid = niftiread([DirData,filesep,name,'_processed.nii']); 
        vid = h5read(data_name,'/mov');
    end
    % get neural traces: demixes the traces for overlapping neurons
    FinalTraces = GetTraces(vid,FinalMasks);
    
    % neuropil correction.
    tag = 1;
    if tag
        FinalTraces = RemoveNeuropil(vid,FinalMasks,FinalTraces,1,0.78);
    end
    
    % Temporal Labeling based on d'
    [Label,~] = TemporalLabeling_d(FinalMasks,FinalTraces,FinalTimes,stat,opt);

    % save result
%     Savename = [DirSave,filesep,'TemporalMask_',name];
%     niftiwrite(Label,[Savename],'Compressed',true);    
    nframes = size(Label,3);
    if isinf(block_size)
        num_blocks = 1;
    else
        num_blocks = ceil(nframes/block_size);
    end
%     if nframes<block_size
%         % imgFinal = int16(imgFinal*128); % YB 2019/12/13
%         niftiwrite(Label,[DirSave,filesep,'TemporalMask_',name],'compressed',true);
%     else
    for kk = 1:num_blocks
        niftiwrite(Label(:,:,((kk-1)*block_size+1):min(kk*block_size,nframes)),...
            [DirSave,'TemporalMask_',name,'part',num2str(kk),'.nii'],'compressed',true);
    end        
%     end
    
end

end

