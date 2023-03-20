function [neuronmasks,times] = piece_neurons_IOU(neuronmasks,thresh_mask,thresh_IOU,times)
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
% area_u = a1 + a2 - area_i;
area_u = area + area' - area_i;

%area_i = area_i+area_i';
%area_u = area_u+area_u';

% IOU = area_i./area_u;
IOU = tril(area_i)./area_u;
[x, y] = find(IOU >= thresh_IOU);
% deleteneurons=unique(x);
belongs=1:N;
% deleteneurons=false(1,n);
% comb=speye(n,n);
for i = 1:length(y)
%     if ~isempty(times{y(i)})
%         neuronmasks(:,x(i)) = neuronmasks(:,x(i))+neuronmasks(:,y(i));
%         neuronmasks(:,y(i)) = 0;
%     %     comb(y(i),y(i))=0;
%     %     comb(y(i),x(i))=1;
%         times{x(i)} = [times{x(i)};times{y(i)}];
%         times{y(i)} = [];
%     %     deleteneurons = [deleteneurons y(i)];
%     %     deleteneurons(y(i))=true;
%     end

    yto=belongs(y(i));
    xto=belongs(x(i));
    if yto ~= xto
        to=min(xto,yto);
        from=max(xto,yto);
        belongs(belongs==from)=to;
        neuronmasks(:,to) = neuronmasks(:,from)+neuronmasks(:,to);
        neuronmasks(:,from) = 0;
    %     comb(y(i),y(i))=0;
    %     comb(y(i),x(i))=1;
        times{to} = [times{from};times{to}];
        times{from} = [];
    %     deleteneurons = [deleteneurons y(i)];
    %     deleteneurons(y(i))=true;
    end
end
% neuronmasks=neuronmasks*comb;
% area(deleteneurons,:) = [];
% neuronmasks(:,deleteneurons) = [];
% times(deleteneurons) = [];
remain=unique(belongs);
neuronmasks=neuronmasks(:,remain);
times=times(remain);
end


% function to = ultimateto(belongs, ind)
% to=belongs(ind);
% while to~=ind
%     ind=to;
%     to=belongs(ind);
% end
% end