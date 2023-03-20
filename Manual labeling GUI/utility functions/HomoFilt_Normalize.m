%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Submitted to PNAS.
%
% Released under a GPL v2 license.

function [imgFinal, num_blocks, time] = HomoFilt_Normalize(vid,DirSave,name,s,NormVals, block_size)
% - vid can be either the directory pointing to the data or the video
% itself (for use within matlab)
% - DirSave is the directory to save processed data
% - name is the name of data used for both read and write
% - s is the standard deviation (in pixels) for the gaussian filter used in the filtering
% process
% - NormVals is 1 1x2 vector of mean and standard deviation to use for data
% normalziation. Set the second element to 0 to use its own standard
% deviation

%%  Initialize missing parameters
if nargin < 6
    block_size = 1e8;
end

if nargin < 5
    NormVals = [0,0];
end

if nargin < 4
    s = 30;
end

if nargin <3
    error('Not enough input variable.')
end

if isstring(name)
    name = char(name);
elseif ~ischar(name)
    error('Wrong format for name.')
end
%% Check which type vid is. If it's a char, then it's pointing to the video 
% file location and name (saved in .nii format)
if isa(vid,'char') || isstring(vid)
    if isstring(vid)
        vid = char(vid);
    end
    if contains(vid,'.nii')
        vid = niftiread(vid);
    elseif contains(vid,'.h5')
        vid = h5read(vid,'/mov');
    end
end

%%
tic;
% Apply homorphic filtering
imgFinal = homo_filt(vid,s);

% Standardize by the standard deviation
imgFinal = imgFinal - NormVals(1);
if NormVals(2)== 0
    stData = std(imgFinal(:));
    imgFinal = imgFinal/stData;
else
    imgFinal = imgFinal/NormVals(2);
end
time = toc

%% save data for later use
if ~exist(DirSave)
    mkdir(DirSave)
end
nframes = size(imgFinal,3);
if isinf(block_size)
    num_blocks = 1;
else
    num_blocks = ceil(nframes/block_size);
end
% if nframes<block_size
%     % imgFinal = int16(imgFinal*128); % YB 2019/12/13
%     niftiwrite(imgFinal,[DirSave,filesep,name,'_dsCropped_HomoNorm'],'compressed',false);
% else
for kk = 1:num_blocks
    niftiwrite(imgFinal(:,:,((kk-1)*block_size+1):min(kk*block_size,nframes)),...
        [DirSave,filesep,name,'part',num2str(kk),'_dsCropped_HomoNorm.nii'],'compressed',false);
end        
% end

end

