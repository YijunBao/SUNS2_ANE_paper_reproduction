load ../fig_initialization/data_BG;
absD = reshape(abs(D),[],size(D,3));
mag = max(absD,[],1)'.*max(F,[],2);
mag_int = round(mag');

figure;
for ii = 1:size(D,3)
    clf;
    imagesc(D(:,:,ii));
    colormap gray;
    colorbar;
    axis('image');
    title(['Component ',num2str(ii)]);
    set(gca,'FontSize',12);
    saveas(gcf,['D_',num2str(ii),'.png']);
end

%%
figure; 
hold on;
F0=F;
F0(end,:)=F(end,:)-1;
[N,T] = size(F);
select = [1,2,9,16];
mag = 10;
for n = 1:4 % select
    plot(F0(select(n),:)*mag+n);
end
legend(arrayfun(@num2str,select,'UniformOutput',false),...
    'Location','NorthEast','NumColumns',2,'FontSize',12);
set(gca,'FontSize',12);
ylim([0,5]);
xticklabels('');
yticklabels('');
saveas(gcf,'Temporal components BG simulation.png')
