%
% Please cite this paper if you use any component of this software:
% S. Soltanian-Zadeh, K. Sahingur, S. Blau, Y. Gong, and S. Farsiu, "Fast 
% and robust active neuron segmentation in two-photon calcium imaging using 
% spatio-temporal deep learning," Proceedings of the National Academy of Sciences (PNAS), 2019.
%
% Released under a GPL v2 license.


function [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks,Masks,Thresh)
NGT = size(GTMasks,2);
NMask = size(Masks,2);

% match Neurons
if isempty(Masks) || isempty(GTMasks)
    Recall = 0;
    Precision = 0;
    F1 = 0;
    m = zeros(NGT,NMask);
else
    Dmat = JaccardDist_2(GTMasks,Masks);
    D = Dmat;
    D(D> 1- Thresh) = Inf;   % no connection between cases with IoU<thresh or equavalantly, dist>1-JaccardThresh
    [m,~] = Hungarian(D);
    num_match=sum(sum(m));
    Recall = num_match/NGT;
    Precision = num_match/NMask;

    F1 = 2*Recall.*Precision./(Recall+Precision);
end
end

