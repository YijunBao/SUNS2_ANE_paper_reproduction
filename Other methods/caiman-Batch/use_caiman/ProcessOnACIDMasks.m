function [Ab] = ProcessOnACIDMasks(Ab,SZ,thresh)
% Ab is the PxN matrix of neuron masks returned by OnACID. It gets
% processed for :
%   - removing low intensity pixels
%   - cutting the 13 pixel borders used in the GT
%   - removing empty components
%
%thresh is used for removing low intensity pixels form neuron masks

%         [B,I] = sort(Ab.^2,1,'ascend');
%         temp = cumsum(B,1);
%         for n = 1:size(Ab,2)
%             ff = find(temp(:,n)>thresh*temp(end,n),1,'first');
%             if ~isempty(ff)
%                 Ab(I(1:ff,n),n) = 0;
%             end
%         end
        % Thresholding based on thr*max(mask)
        MaxA = max(Ab,[],1);
        Ab = Ab >= thresh*MaxA;
        Ab = reshape(Ab,SZ(2),SZ(1),[])>0;
        Ab = permute(Ab,[2,1,3]);   %Needs transpose to be correct


        ind = find(sum(sum(Ab,1),2)==0);       
        Ab(:,:,ind) = [];
% %         %Fill holes induced by thresholding
% %         for k = 1:size(Ab,3)
% %             Ab(:,:,k) = imfill(Ab(:,:,k),'holes');
% %         end
% %         
%         if SZ(1) == 512
%             Ab = Ab(13:end-13,13:end-13,:);
%             
%             M = ones(487,487); M(2:end-1,2:end-1) = 0; 
%             indBorder = find(sum(reshape(Ab,[],size(Ab,3)).*M(:),1));
%             
%             ind = find(sum(sum(Ab(:,:,indBorder),1),2)<=30);
%             Ab(:,:,indBorder(ind)) = [];
% %         % remove empty masks
%             ind = find(sum(sum(Ab,1),2)==0);
%             Ab(:,:,ind) = [];
%         end
end

