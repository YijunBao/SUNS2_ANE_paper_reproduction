%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Submitted to PNAS.
%
% Released under a GPL v2 license.

function SaveSubVolumes(DataDir,MaskDir,DirSave,name,Ws,Wt, num_blocks)
% Ws ans Wt are the spatial and temporal window lenghts

% create the candidate window locations with %50 overlap spatially, 75%
% temporally
stepS = round(Ws/2); stepT = round(Wt/4); %/3
% stepT = Wt;
% ns = ceil(487/Ws);
% stepS = ceil((487-Ws)/(ns-1)); 

% if num_blocks == 1 && exist([MaskDir,filesep,'TemporalMask_',name,'.nii.gz']) ...
%         && exist([DataDir,filesep,name,'_dsCropped_HomoNorm.nii']) %.gz
%     
%     label = niftiread([MaskDir,filesep,'TemporalMask_',name,'.nii.gz']); %
%     vid = niftiread([DataDir,filesep,name,'_dsCropped_HomoNorm.nii']); %.gz
%     
%     [x,y,T] = size(vid);
%     ns = ceil(max(x,y)/Ws);
%     stepS = ceil((max(x,y)-Ws)/(ns-1)); 
%     rp = [1:stepS:x-Ws-1, x-Ws]; %last element is to cover all the image
%     cp = [1:stepS:y-Ws-1, y-Ws]; 
%     tp = [1:stepT:T-Wt];
%     fprintf('%d*%d*%d*3\n', length(rp), length(cp), length(tp));
%     
%     count = 1;
%     nameVid = ['HomoVid_',name];
%     nameMask = ['TempMask_',name];
%     for jj = 0:2
%         fprintf('Rotate %d degrees\n', 90*jj);
%         count = randWinSave(rot90(vid,jj),rot90(label,jj),...
%              DirSave,nameVid,nameMask,Ws,Wt,rp,cp,tp,count,jj);
%     end
%     clear vid label
if exist([MaskDir,filesep,'TemporalMask_',name,'part1.nii.gz']) ...
        && exist([DataDir,filesep,name,'part1_dsCropped_HomoNorm.nii']) %.gz
    
    for kk = 1:num_blocks
        label = niftiread([MaskDir,filesep,'TemporalMask_',name,'part',num2str(kk),'.nii.gz']); %
        vid = niftiread([DataDir,filesep,name,'part',num2str(kk),'_dsCropped_HomoNorm.nii']); %.gz

        [x,y,T] = size(vid);
        ns = ceil(max(x,y)/Ws);
        stepS = ceil((max(x,y)-Ws)/(ns-1)); 
        rp = [1:stepS:x-Ws-1, x-Ws]; %last element is to cover all the image
        cp = [1:stepS:y-Ws-1, y-Ws]; 
        tp = [1:stepT:T-Wt];
%         tp = [1000:stepT:T-Wt];
        fprintf('part%d: %d*%d*%d*3\n', kk, length(rp), length(cp), length(tp));

        count = 1;
        nameVid = ['HomoVid_',name,'part',num2str(kk)];
        nameMask = ['TempMask_',name,'part',num2str(kk)];
        for jj = 0:2
            fprintf('Rotate %d degrees\n', 90*jj);
            count = randWinSave(rot90(vid,jj),rot90(label,jj),...
                 DirSave,nameVid,nameMask,Ws,Wt,rp,cp,tp,count,jj);
        end
        clear vid label
    end
else
    error('Normalized data or Mask not found.')
end

end

