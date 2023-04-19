function overlap = overlap_neurons_consume(neuronmasks,avgA,thresh_mask,thresh_consume,times)
% Combine masks with large IOU or consume.

% thresh_IOU=0.5;
% thresh_consume=0.8;
% thresh_consume should be larger than 2*thresh_IOU/(1+thresh_IOU) to avoid
    % two masks consuming each other.

% [Lx,Ly,n] = size(neuronmasks);
N=size(neuronmasks,2);
% neuronmasks=sparse(reshape(neuronmasks,Lx*Ly,n));
tempmasks=double(neuronmasks>=thresh_mask*max(neuronmasks,[],1));
% area = zeros(n,n);
% area_i = zeros(n,n);
% area_u = zeros(n,n);
area = full(sum(tempmasks,1))'; %area of i^th neuron
% for i = 1:n-1
% %     area_i(i,i+1:n) = reshape(sum(tempmasks(:,i) & tempmasks(:,i+1:n)),1,n-i);
% %     area_u(i,i+1:n) = reshape(sum(tempmasks(:,i) | tempmasks(:,i+1:n)),1,n-i);
%     area_i(i,i+1:n) = sum(tempmasks(:,i) & tempmasks(:,i+1:n));
%     area_u(i,i+1:n) = sum(tempmasks(:,i) | tempmasks(:,i+1:n));
% end
area_i=full(tempmasks'*tempmasks);
area_i=area_i-diag(area);
% for ii=1:n
%     area_i(ii,ii)=0;
% end
% [a1,a2]=meshgrid(area,area);
% a1=repmat(area,1,N);
% a2=repmat(area',N,1);

%area_i = area_i+area_i';
%area_u = area_u+area_u';

% consume = (area_i+area_i')./area;
consume = area_i./area;
[x, y] = find(consume >= thresh_consume);
overlap = [x, y]; 
