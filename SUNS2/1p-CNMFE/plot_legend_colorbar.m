color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% green = [0.1,0.9,0.1]; % color(5,:); %
% red = [0.9,0.1,0.1]; % color(7,:); %
% blue = [0.1,0.8,0.9]; % color(6,:); %
yellow = [0.8,0.8,0.0]; % color(3,:); %
magenta = [0.9,0.3,0.9]; % color(4,:); %
green = [0.0,0.65,0.0]; % color(5,:); %
red = [0.8,0.0,0.0]; % color(7,:); %
blue = [0.0,0.6,0.8]; % color(6,:); %


%% Plot 2 color legend
figure('Position',[100,100,800,200]);
hold on;
plot(1,1,'o','Color',color(3,:),'MarkerSize',20,'LineWidth',2);
plot(2,1,'o','Color',red,'MarkerSize',20,'LineWidth',2);
xlim([0.5,4])
text(1.14,1,'Ground Truth','Fontsize',14,'FontName','Arial');
text(2.14,1,'Computer Segmentation','Fontsize',14,'FontName','Arial');
saveas(gcf,'n color legend.emf');

%% Plot 3 color legend
figure('Position',[100,100,800,200]);
hold on;
plot(1,1,'o','Color',green,'MarkerSize',24,'LineWidth',2);
plot(2,1,'o','Color',red,'MarkerSize',24,'LineWidth',2);
plot(3,1,'o','Color',blue,'MarkerSize',24,'LineWidth',2);
xlim([0.5,4])
text(1.2,1,'True positive','Fontsize',12,'FontName','Arial');
text(2.2,1,'False positive','Fontsize',12,'FontName','Arial');
text(3.2,1,'False negative','Fontsize',12,'FontName','Arial');
saveas(gcf,'3 color legend.svg');

%% Plot 3 color legend 2 lines
figure('Position',[100,100,800,200]);
hold on;
plot(1,1,'o','Color',green,'MarkerSize',24,'LineWidth',2);
plot(1.8,1,'o','Color',red,'MarkerSize',24,'LineWidth',2);
plot(2.6,1,'o','Color',blue,'MarkerSize',24,'LineWidth',2);
xlim([0.5,4])
text(1.2,1,{'True','positive'},'Fontsize',14,'FontName','Arial');
text(2.0,1,{'False','positive'},'Fontsize',14,'FontName','Arial');
text(2.8,1,{'False','negative'},'Fontsize',14,'FontName','Arial');
saveas(gcf,'3 color legend 2 lines.svg');


%% A horizontal colorbar
figure('Position',[50,50,200,300],'Color','w');
imagesc(ones(24,24),[200,800]); axis('image'); colormap gray; %
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','Peak Intensity');
set(h,'Location','Southoutside','FontSize',21,'Ticks',[200,500,800]);
saveas(gcf,'horizontal colorbar.svg');
saveas(gcf,'horizontal colorbar.png');

%% A vertical colorbar
figure('Position',[50,50,300,200],'Color','w');
% colorbar_range = [2,14]; % 
% colorbar_range = [0,10]; % 
% colorbar_range = [0,5]; % 
% colorbar_range = [2,6]; % 
colorbar_range = [0,8]; % 
imagesc(ones(24,24),colorbar_range); axis('image'); colormap gray; %
xticklabels({}); yticklabels({});
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
% set(h,'Location','Southoutside','FontSize',14,'Ticks',2:2:14);
% set(h,'FontSize',14,'Ticks',2:2:14);
% set(h,'FontSize',14,'Ticks',0:5:10);
% set(h,'FontSize',14,'Ticks',0:1:5);
set(h,'FontSize',14,'Ticks',2:1:6);
saveas(gcf,['vertical colorbar ',mat2str(colorbar_range),'.emf']);
saveas(gcf,['vertical colorbar ',mat2str(colorbar_range),'.svg']);
saveas(gcf,['vertical colorbar ',mat2str(colorbar_range),'.png']);

