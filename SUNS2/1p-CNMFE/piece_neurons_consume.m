function [neuronmasks,times] = piece_neurons_consume(neuronmasks,avgA,thresh_mask,thresh_consume,times)
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
%%
% deleteneurons=false(1,N);
belongs=1:N;
throw=2*N+1;
% comb=speye(n,n);
[x, y] = find(consume >= thresh_consume);
for i = 1:length(x)
%     if y(i) > avgA
% %         deleteneurons = [deleteneurons y(i)];
%         deleteneurons(y(i))=true;
%     %     comb(y(i),y(i))=0;
%     else
% %         deleteneurons = [deleteneurons x(i)];
%         deleteneurons(x(i))=true;
%         neuronmasks(:,y(i)) = neuronmasks(:,x(i))+neuronmasks(:,y(i));
%         neuronmasks(:,x(i)) = 0;
%     %     comb(x(i),x(i))=0;
%     %     comb(x(i),y(i))=1;
%         times{y(i)} = [times{x(i)};times{y(i)}];
%         times{x(i)} = [];
%     end

    area_x = area(x(i));
    area_y = area(y(i));
    if max(area_x,area_y) > avgA
        if area_y > area_x
            belongs(y(i))=throw;
            neuronmasks(:,y(i)) = 0;
            times{y(i)} = [];
        else
            belongs(x(i))=throw;
            neuronmasks(:,x(i)) = 0;
            times{x(i)} = [];
        end
    else
        yto=belongs(y(i));
        xto=belongs(x(i));
        if xto ~= yto
            to=min(xto,yto);
            from=max(xto,yto);
            belongs(belongs==from)=to;
            neuronmasks(:,to) = neuronmasks(:,from)+neuronmasks(:,to);
            neuronmasks(:,from) = 0;
            times{to} = [times{from};times{to}];
            times{from} = [];
        end
    end
end
remain=setdiff(belongs,throw);
neuronmasks=neuronmasks(:,remain);
times=times(remain);
% % neuronmasks=neuronmasks*comb;
% neuronmasks(:,deleteneurons) = [];
% times(deleteneurons) = [];

