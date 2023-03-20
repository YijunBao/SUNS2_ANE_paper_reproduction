function [traces]=generate_traces_from_masks_mm(mm,masks)
% Generate background traces for each neuron from ground truth masks
[Lx,Ly,ncells]=size(masks);
T=mm.Format{2}(2);

% [xx, yy] = meshgrid(1:Ly,1:Lx); 
% r_bg=sqrt(mean(sum(sum(masks)))/pi)*2.5;

% if Lx==Lxm && Ly==Lym
%     video=reshape(video,[Lxm*Lym,T]);
% else
%     video=reshape(video(floor((Lx-Lxm)/2)+1:floor((Lx+Lxm)/2),floor((Ly-Lym)/2)+1:floor((Ly+Lym)/2),:),[Lxm*Lym,T]);
% end

traces=zeros(ncells,T,'single');
% bgtraces=zeros(ncells,T,'single');
parfor nn=1:ncells
    mask = masks(:,:,nn);
%     [xxs,yys]=find(mask>0);
%     comx=mean(xxs);
%     comy=mean(yys);
%     circleout = (yy-comx).^2 + (xx-comy).^2 < r_bg^2; 
%     B_temp=mm.Data.B(circleout(:),:);
%     bgtraces(nn,:)=median(B_temp,1);
%     bgtraces(nn,:)=median(mm.Data.B(circleout(:),:),1);

    mask=reshape(mask,[],1);
    video_temp=mm.Data.video(mask(:),:);
    traces(nn,:)=mean(video_temp,1);
end


