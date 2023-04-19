function [Dmat] = JaccardDist_2(M1,M2)

N1 = size(M1,2);
N2 = size(M2,2);

% M1 = sparse(reshape(M1,[],N1));
% M2 = sparse(reshape(M2,[],N2));

% A1 = M1'*M1; a1 = repmat(diag(A1),1,N2);
% A2 = M2'*M2; a2 = repmat(diag(A2)',N1,1);
a1=repmat(sum(M1,1)',1,N2);
a2=repmat(sum(M2,1),N1,1);

intersectMat = M1'*M2;
unionMat = a1 + a2 - intersectMat;
JaccardInd = intersectMat./unionMat;

Dmat = 1-JaccardInd;
% Dmat(Dmat==1) = Inf; %this will get set in the GtePerofmance_Jaccard code

%added condition: If one mask is the subset of the other,
%distnce is zero
% indSubset = find(intersectMat==a1 | intersectMat==a2);
% Dmat(indSubset) = 0;

end

