color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
green = [0.1,0.9,0.1];
red = [0.9,0.1,0.1];
blue = [0.1,0.8,0.9];
yellow = [0.9,0.9,0.1];
magenta = [0.9,0.3,0.9];
colors = distinguishable_colors(16);


%% Figure 2C:
% F1 and speed (log) for all methods for simulated videos
data_name = 'lowBG=5e+03,poisson=1 cv 1025';
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = [1,2,6,3,4]; % 1:4;
% list_method = {'MIN1PIPE', 'CNMF-E', 'SUNS1', 'SUNS2', 'SUNS1-MF'};
list_method = list_method(select);
F1 = F1(:,select);
Speed = Speed(:,select);

num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,400,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot(10*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 1.05+(0:3)*0.05;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
% text(fps_mean(3),list_y_star(3)+0.03,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));

list_x_line = 10.^(3.2+(0:3)*0.2);
list_x_star = list_x_line*1.4;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.2]);
xlim([1,10000]);
% xlim([0.1,1000]);
xticks(10.^(-1:4));
set(gca,'xticklabel',get(gca,'xtick'));
set(gca,'xticklabelRotation',0);
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
saveas(gcf,['Fig2C - F1 speed ',data_name,'.emf']);
saveas(gcf,['Fig2C - F1 speed ',data_name,'.png']);


%% Figure 3E:
% F1 and speed (log) for all methods for TENASPIS videos
data_name = 'TENASPIS_refined_9par 1025'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = [1,2,9,3,4,5];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,500],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.95,1.00,1.05,1.10,1.15]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
line([fps_mean(5),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(5,:));

list_x_line = [0.2,0.3,320,480,720]; % 10.^(2-(4:-1:0)*0.15); % 
list_x_star = list_x_line*1.3;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);
line(list_x_line(5)*[1,1],[F1_mean(5),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1.01,F1_mean(5),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(5,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.2]);
% xlim([10,10000]);
xlim([0.1,1000]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
set(gca,'xticklabelRotation',0);
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['Fig3E - F1 speed ',data_name,'.emf']);
saveas(gcf,['Fig3E - F1 speed ',data_name,'.png']);


%% Figure S9D: 
% F1 and speed for SUNS1 and SUNS2 with or without SF for TENASPIS videos
data_name = 'TENASPIS_refined_9par 1025'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE','SUNS1 (no SF)','SUNS2 (no SF)'};
select = [6,3,7,4];
Recall = Recall(:,select);
Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);

num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
% colors = distinguishable_colors(14);
colors_select = [colors(8,:);color(4,:);colors(7,:);color(5,:)];
% colors_select = color([1,2,4:num_method+1],:);
% colors_select(3,:)=(1+colors(3,:))/2;
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
% plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]
% 
list_y_line = 0.95+(0:5)*0.07;
list_y_star = list_y_line+0.015;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(3)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1)+0.03,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',colors_select(1,:));
% text(fps_mean(1),list_y_star(1),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(1),fps_mean(2)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(3),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(3),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(2),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(2),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(1)+0.03,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',colors_select(2,:));
line([fps_mean(1),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(4),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:));
% 
list_x_line = 298-(0:5)*9; % [4,6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line+2;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(2)*[1,1],[F1_mean(1),F1_mean(3)],'color','k','LineWidth',2)
text(list_x_star(2)*1,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:),'Rotation',270);
line(list_x_line(3)*[1,1],[F1_mean(1),F1_mean(2)],'color','k','LineWidth',2)
text(list_x_star(3)*1,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:),'Rotation',270);
line(list_x_line(4)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1,F1_mean(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:),'Rotation',270);
line(list_x_line(5)*[1,1],[F1_mean(2)-0.01,F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1+4,F1_mean(2),'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',colors_select(2,:),'Rotation',270);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:),'Rotation',270);

% legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

% set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.2]);
% xlim([10,10000]);
% xlim([100,1000]);
% xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS9D - F1 speed ',data_name(1:end-8),' noSF.emf']);
saveas(gcf,['FigS9D - F1 speed ',data_name(1:end-8),' noSF.png']);


%% Figure S10B:
% F1 and speed (log) for all methods with ANE for TENASPIS videos
data_name = 'TENASPIS_ANE_5 1026'; % 'TENASPIS_original'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
list_method = cellfun(@(x)[x,'-ANE'],list_method,'UniformOutput',false);
num_method = size(F1,2);
select = [1,2,5,3,4];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(3,:)=(1+colors_select(3,:))/2;
% colors_select(2,:)=(colors_select(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.95,1.00,1.05,1.1]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.008;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(4)+0.025,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',colors_select(4,:));

list_x_line = 10.^(1.9-(3:-1:0)*0.15); % [4,6,60,90]; % 
list_x_star = list_x_line*1.25;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1,F1_mean(5),'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',colors_select(3,:),'Rotation',270);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1,F1_mean(5),'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',colors_select(4,:),'Rotation',270);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.12]);
% xlim([10,10000]);
xlim([0.1,100]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS10B - F1 speed ',data_name,'.emf']);
saveas(gcf,['FigS10B - F1 speed ',data_name,'.png']);


%% Figure S11D:
% F1 and speed (log) for all methods starting from the SNR videos for TENASPIS videos
data_name = 'TENASPIS_SNR'; % 'TENASPIS_original'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
% list_method = cellfun(@(x)[x,'-ANE'],list_method,'UniformOutput',false);
num_method = size(F1,2);
select = [1,2,6,3,4,5];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(3,:)=(1+colors_select(3,:))/2;
% colors_select(2,:)=(colors_select(2,:))/2;
colors_select(3,:)=(colors_select(3,:))/2;
colors_select(2,:)=(0.5+colors_select(2,:))/1.5;
colors_select(1,:)=(1+colors_select(1,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 0.95+(0:5)*0.05;
list_y_star = list_y_line+0.008;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
line([fps_mean(5),fps_mean(end)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(5),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(5,:));

list_x_line = 12.^(2-(4:-1:0)*0.15); % [0.2,0.3,320,480,720]; % 
list_x_star = list_x_line*1.3;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);
line(list_x_line(5)*[1,1],[F1_mean(5),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1.01,F1_mean(5),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(5,:),'Rotation',90);

legend([list_method,['Video']],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % ,newline,'rate'
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.2]);
% xlim([10,10000]);
xlim([0.1,1000]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS11D - F1 speed ',data_name,'.emf']);
saveas(gcf,['FigS11D - F1 speed ',data_name,'.png']);


%% Figure S11E:
% Change of F1 and speed (log) for three other methods starting from the SNR videos for TENASPIS videos
data_name_SNR = 'TENASPIS_SNR'; %
SNR = load(['F1 speed ',data_name_SNR,'.mat'],'Recall','Precision','F1','Speed','list_method');
data_name_raw = 'TENASPIS_refined_9par 1025';
raw = load(['F1 speed ',data_name_raw,'.mat'],'Recall','Precision','F1','Speed','list_method');
data_name = 'TENASPIS_raw_SNR';
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
% list_method = cellfun(@(x)[x,'-ANE'],list_method,'UniformOutput',false);
num_method = size(SNR.F1,2);
select_SNR = [1,2,6];
select_raw = [1,2,9];
list_method_raw = SNR.list_method(:,select_SNR);
list_method_SNR = cellfun(@(x) [x,' SNR'],list_method_raw,'UniformOutput',false);
list_method = [list_method_raw,list_method_SNR];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = [raw.F1(:,select_raw), SNR.F1(:,select_SNR)];
Speed = [raw.Speed(:,select_raw), SNR.Speed(:,select_SNR)];
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
colors_select(6,:)=(colors_select(3,:))/2;
colors_select(5,:)=(0.5+colors_select(2,:))/1.5;
colors_select(4,:)=(1+colors_select(1,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;

line([fps_mean(1),fps_mean(4)],[F1_mean(1),F1_mean(4)],'LineStyle','--','color',colors_select(4,:),'LineWidth',1,'HandleVisibility','off')
line([fps_mean(2),fps_mean(5)],[F1_mean(2),F1_mean(5)],'LineStyle','--','color',colors_select(5,:),'LineWidth',1,'HandleVisibility','off')
line([fps_mean(3),fps_mean(6)],[F1_mean(3),F1_mean(6)],'LineStyle','--','color',colors_select(6,:),'LineWidth',1,'HandleVisibility','off')

% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 0.85+(0:5)*0.05;
list_y_star = list_y_line+0.008;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(4)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(1),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(6)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(6),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));

list_x_line = 12.^(1.4-(4:-1:0)*0.15); % [0.2,0.3,320,480,720]; % 
list_x_star = list_x_line*1.2;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(4)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(6)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(6),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);

legend([list_method,['Video']],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % ,newline,'rate'
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.2]);
% xlim([10,10000]);
xlim([0.1,100]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS11E - F1 speed ',data_name,'.emf']);
saveas(gcf,['FigS11E - F1 speed ',data_name,'.png']);


%% Figure S12B:
% F1 and speed (log) for all methods for TENASPIS videos using initial GT masks
data_name = 'TENASPIS_original_6 1026'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = [1,2,6,3,4,5];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 0.9+(0:4)*0.05; % [0.95,1.03,0.99,1.07,1.11]; % 
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
line([fps_mean(5),fps_mean(end)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(5),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(5,:));

list_x_line = [50,75,400,600,900]; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line*1.3;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);
line(list_x_line(5)*[1,1],[F1_mean(5),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1.01,F1_mean(5),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(5,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.12]);
% xlim([10,10000]);
xlim([0.1,1000]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS12B - F1 speed ',data_name,'.emf']);
saveas(gcf,['FigS12B - F1 speed ',data_name,'.png']);


%% Figure S13D:
% F1 and speed (log) for DeepWonder, SUNS2, and SUNS2-ANE for TENASPIS videos
data_name = 'TENASPIS_refined_9par 1025'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = [8,4,5];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors = distinguishable_colors(16);
colors_select = [colors(8,:); color([5,6],:)];

% %%
figure('Position',[50,50,400,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.93,0.98]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(2)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(1),fps_mean(3)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));

list_x_line = [20,30,60,90]*0.3; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line*0.93;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(2)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(1),F1_mean(3)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
% line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
% line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.0]);
% xlim([10,10000]);
xlim([1,1000]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS13D - F1 speed ',data_name,' DeepWonder.emf']);
saveas(gcf,['FigS13D - F1 speed ',data_name,' DeepWonder.png']);


%% Figure S14D:
% F1 and speed (log) for SUNS1, SUNS2, and CNMF for TENASPIS videos
data_name = 'TENASPIS CaImAn'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = [1,2,4];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors = distinguishable_colors(21);
colors_select(1,:)=colors(21,:);
colors_select(3,:)=color(5,:);
% colors_select(4,:)=color(5,:);
% colors_select(5,:)=color(6,:);
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',16, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.85,0.90,0.95,1.00]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(2)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(1),fps_mean(3)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
% line([fps_mean(4),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(4),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
% line([fps_mean(5),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(5),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(5,:));

list_x_line = [320,480,720]; % 10.^(2-(4:-1:0)*0.15); % 
list_x_star = list_x_line*1.3;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(2)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(1),F1_mean(3)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
% line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(3)*1.01,F1_mean(3),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
% line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(4)*1.01,F1_mean(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);
% line(list_x_line(5)*[1,1],[F1_mean(5),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(5)*1.01,F1_mean(5),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(5,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',16,'NumColumns',3); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.0]);
% xlim([10,10000]);
xlim([1,1000]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
set(gca,'xticklabelRotation',0);
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['FigS14D - F1 speed ',data_name,'.emf']);
saveas(gcf,['FigS14D - F1 speed ',data_name,'.png']);


%% Figure 4D and Figure S19D:
% F1 and speed (log) for all methods for CNMF-E videos
rate_hz = [10,15,15,5]; % frame rate of each video
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_9 1025'];
    load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
    select = [1,2,9,3,4,5]; % 1:5;
    % Recall = Recall(:,select);
    % Precision = Precision(:,select);
    F1 = F1(:,select);
    Speed = Speed(:,select);
    list_method = list_method(:,select);
    % list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE','Video rate'};
    list_method = [list_method,'Video rate'];
    num_method = size(F1,2);
    % F1=F1(:,[2,5,4,6]);
    % Speed=Speed(:,[2,5,4,6]);
    F1_mean = F1(end-1,:);
    F1_std = F1(end,:);
    fps_mean = Speed(end-1,:);
    fps_std = Speed(end,:);
    F1=F1(1:end-2,:);
    Speed=Speed(1:end-2,:);

    n_colors = length(F1_mean);
    colors_select = color(1:num_method,:);
    colors_select(3,:)=colors(14,:);
    % colors_select(2,:)=(colors(2,:))/2;

    % %%
    % figure('Position',[50,50,450,500],'color','w');
    figure('Position',[50,50,450,450],'color','w');
    hold on;
    % errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
    for k = 1:n_colors
        plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
    end
    for k = 1:n_colors
        errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),min(fps_mean(k)-1,fps_std(k)),fps_std(k),...
            'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
    end
    plot(rate_hz(data_ind)*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
    xlabel('Speed (Frame/s)');
    ylabel('{\itF}_1');
    % set(gca,'FontName','Arial','FontSize',18, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]
    if data_ind==3
        set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]
    else
        set(gca,'FontName','Arial','FontSize',18, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]
    end

    % list_y_line = [0.93,0.97,0.95,0.99]; % 0.99-(3:-1:0)*0.04;
    % list_y_star = list_y_line+0.005;
    % % reorder = [1,2,6,3,4,5];
    % % list_y_line = list_y_line(reorder);
    % % list_y_star = list_y_star(reorder);
    % line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
    % line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
    % 
    % list_x_line = [0.4,0.6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
    % list_x_star = list_x_line*1.3;
    % % reorder = [1,5,2,6,4,3];
    % % list_x_line = list_x_line(reorder);
    % % list_x_star = list_x_star(reorder);
    % line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
    % line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
    % line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
    % line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);

    if data_ind==3
        legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % 
    else
        legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',18,'NumColumns',3); % 
    end
    % legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',18,'NumColumns',length(list_method)); %
    % legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

    set(gca, 'XScale', 'log');
    % yticks(0.6:0.1:1)
    % ylim([0.1,0.8]);
    ylim([0,1]);
    % xlim([1,1000]);
    if data_ind == 1
        xlim([1,10000]);
    elseif data_ind == 3
        xlim([10,10000]);
    else
        xlim([1,1000]);
    end
    xticks(10.^(0:4));
    set(gca,'xticklabel',get(gca,'xtick'));
    set(gca,'xticklabelRotation',0);
    box off
    two_errorbar_position = get(gca,'Position');
    % title('Cropped one-photon videos');
    % saveas(gcf,['F1 speed ',data_name,'.emf']);
    % saveas(gcf,['F1 speed ',data_name,'.png']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),data_name(end-4:end),'.emf']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),data_name(end-4:end),'.png']);
    saveas(gcf,['Fig4D ',num2str(data_ind),'- F1 speed ',data_name(1:end-7),'.emf']);
    saveas(gcf,['Fig4D ',num2str(data_ind),'- F1 speed ',data_name(1:end-7),'.png']);
    if data_ind == 3
        saveas(gcf,['FigS19D - F1 speed ',data_name(1:end-7),'.emf']);
        saveas(gcf,['FigS19D - F1 speed ',data_name(1:end-7),'.png']);
    end
end


%% Figure S16B:
% F1 and speed for SUNS1 and SUNS2 with or without SF for CNMF-E videos
rate_hz = [10,15,15,5]; % frame rate of each video
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_9 1025'];
    load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
    % list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE','SUNS1 (no SF)','SUNS2 (no SF)'};
    select = [6,3,7,4];
    % Recall = Recall(:,select);
    % Precision = Precision(:,select);
    F1 = F1(:,select);
    Speed = Speed(:,select);
    list_method = list_method(:,select);

    num_method = size(F1,2);
    F1_mean = F1(end-1,:);
    F1_std = F1(end,:);
    fps_mean = Speed(end-1,:);
    fps_std = Speed(end,:);
    F1=F1(1:end-2,:);
    Speed=Speed(1:end-2,:);

    n_colors = length(F1_mean);
    % colors = distinguishable_colors(14);
    colors_select = [colors(8,:);color(4,:);colors(7,:);color(5,:)];
    % colors_select(3,:)=(1+colors(3,:))/2;
    % colors_select(2,:)=(colors(2,:))/2;

    % %%
    figure('Position',[50,50,400,420],'color','w');
    hold on;
    % errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
    for k = 1:n_colors
        plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
    end
    for k = 1:n_colors
        errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),min(fps_mean(k)-1,fps_std(k)),fps_std(k),...
            'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
    end
    % plot(rate_hz(data_ind)*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
    xlabel('Speed (Frame/s)');
    ylabel('{\itF}_1');
    set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

    % list_y_line = [0.93,0.97,0.95,0.99]; % 0.99-(3:-1:0)*0.04;
    % list_y_star = list_y_line+0.005;
    % % reorder = [1,2,6,3,4,5];
    % % list_y_line = list_y_line(reorder);
    % % list_y_star = list_y_star(reorder);
    % line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
    % line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
    % 
    % list_x_line = [0.4,0.6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
    % list_x_star = list_x_line*1.3;
    % % reorder = [1,5,2,6,4,3];
    % % list_x_line = list_x_line(reorder);
    % % list_x_star = list_x_star(reorder);
    % line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
    % line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
    % line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
    % line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);

    legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
    % legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

    % set(gca, 'XScale', 'log');
    % yticks(0.6:0.1:1)
    % ylim([0.1,0.8]);
    ylim([0,1]);
    % xlim([0,4000]);
    % xlim([0,600]);
    % xlim([0,4000]);
    % xlim([0,1200]); xticks(0:400:1200);
    if data_ind == 2
        xlim([0,600]);
    elseif data_ind == 4
        xlim([0,1200]); xticks(0:400:1200);
    else
        xlim([0,4000]);
    end
    % xticks(10.^(0:4));
    set(gca,'xticklabel',get(gca,'xtick'));
    box off
    % title('Cropped one-photon videos');
    % saveas(gcf,['F1 speed ',data_name(1:end-7),' noSF',data_name(end-4:end),'.emf']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),' noSF',data_name(end-4:end),'.png']);
    saveas(gcf,['FigS16B-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' noSF.emf']);
    saveas(gcf,['FigS16B-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' noSF.png']);
end


%% Figure S18D:
% F1 and speed for DeepWonder, SUNS2, and SUNS2-ANE for CNMF-E videos
rate_hz = [10,15,15,5]; % frame rate of each video
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_9 1025'];
    load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
    % list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE','SUNS1 (no SF)','SUNS2 (no SF)'};
    select = [8,4,5];
    % Recall = Recall(:,select);
    % Precision = Precision(:,select);
    F1 = F1(:,select);
    Speed = Speed(:,select);
    list_method = list_method(:,select);
    list_method = [list_method,'Video rate'];

    num_method = size(F1,2);
    F1_mean = F1(end-1,:);
    F1_std = F1(end,:);
    fps_mean = Speed(end-1,:);
    fps_std = Speed(end,:);
    F1=F1(1:end-2,:);
    Speed=Speed(1:end-2,:);

    n_colors = length(F1_mean);
    colors = distinguishable_colors(14);
    colors_select = [colors(8,:);color(5,:);color(6,:)];
    % colors_select(3,:)=(1+colors(3,:))/2;
    % colors_select(2,:)=(colors(2,:))/2;

    % %%
    figure('Position',[50,50,400,450],'color','w');
    hold on;
    % errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
    for k = 1:n_colors
        plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
    end
    for k = 1:n_colors
        errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),min(fps_mean(k)-1,fps_std(k)),fps_std(k),...
            'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
    end
    plot(rate_hz(data_ind)*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
    xlabel('Speed (Frame/s)');
    ylabel('{\itF}_1');
    set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]
    set(gca,'xticklabelRotation',0);

    % list_y_line = [0.93,0.97,0.95,0.99]; % 0.99-(3:-1:0)*0.04;
    % list_y_star = list_y_line+0.005;
    % % reorder = [1,2,6,3,4,5];
    % % list_y_line = list_y_line(reorder);
    % % list_y_star = list_y_star(reorder);
    % line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
    % line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
    % 
    % list_x_line = [0.4,0.6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
    % list_x_star = list_x_line*1.3;
    % % reorder = [1,5,2,6,4,3];
    % % list_x_line = list_x_line(reorder);
    % % list_x_star = list_x_star(reorder);
    % line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
    % line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
    % line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
    % line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);

    legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
    % legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

    set(gca, 'XScale', 'log');
    % yticks(0.6:0.1:1)
    % ylim([0.1,0.8]);
    ylim([0,1]);
    % xlim([0,4000]);
    % xlim([0,600]);
    % xlim([0,4000]);
    % xlim([0,1200]); xticks(0:400:1200);
    if data_ind == 1
        xlim([0.1,10000]);
    elseif data_ind == 3
        xlim([1,10000]);
    else
        xlim([1,1000]);
    end
    xticks(10.^(-1:4));
    set(gca,'xticklabel',get(gca,'xtick'));
    box off
    % title('Cropped one-photon videos');
    % saveas(gcf,['F1 speed ',data_name(1:end-7),' noSF',data_name(end-4:end),'.emf']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),' noSF',data_name(end-4:end),'.png']);
    saveas(gcf,['FigS18D-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' DeepWonder.emf']);
    saveas(gcf,['FigS18D-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' DeepWonder.png']);
end


%% Figure 5D spatial:
% F1 and speed (log) for all methods for TUnCaT videos
data_name = 'temporal data_TUnCaT_7 weight tol=1e-4_nbin=2 1024'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = [1,2,7,3,4,5];
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,500],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('Spatial {\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 0.85:0.05:1.05; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
line([fps_mean(5),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(5,:));

list_x_line = 10.^(4-(4:-1:0)*0.1); % [0.2,0.3,320,480,720]; % 
list_x_star = list_x_line*1.2;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(4,:),'Rotation',90);
line(list_x_line(5)*[1,1],[F1_mean(5),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1.01,F1_mean(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors_select(5,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.1]);
xlim([10,10000]);
xticks(10.^(1:4));
set(gca,'xticklabel',get(gca,'xtick'));
set(gca,'xticklabelRotation',0);
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['Fig5D - F1 speed spatial ',data_name,'.emf']);
saveas(gcf,['Fig5D - F1 speed spatial ',data_name,'.png']);


%% Figure 5F temporal:
% F1 and speed (log) for all methods for TUnCaT videos
data_name = 'temporal data_TUnCaT_7 weight tol=1e-4_nbin=2 1024'; % 
load(['F1 speed ',data_name,'.mat'],'Recall_temporal','Precision_temporal','F1_temporal','Speed_temporal','list_method');
select = [1,2,7,3,4,5];
% Recall = Recall_temporal(:,select);
% Precision = Precision_temporal(:,select);
F1 = F1_temporal(:,select);
Speed = Speed_temporal(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
num_method = size(F1,2);
% F1=F1(:,[2,5,4,6]);
% Speed=Speed(:,[2,5,4,6]);
F1_mean = F1(end-1,:);
F1_std = F1(end,:);
fps_mean = Speed(end-1,:);
fps_std = Speed(end,:);
F1=F1(1:end-2,:);
Speed=Speed(1:end-2,:);

n_colors = length(F1_mean);
colors_select = color(1:num_method,:);
colors_select(3,:)=colors(14,:);
% colors_select(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,500],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors_select(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors_select(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('Weighted spatiotemporal {\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 0.9:0.05:1.15; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(5),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(4)+0.018,'n.s.','HorizontalAlignment', 'left','FontSize',11,'Color',colors_select(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(3,:));
line([fps_mean(4),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(4,:));
line([fps_mean(5),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(5),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors_select(5,:));

list_x_line = 10.^(3-(4:-1:0)*0.09); % [0.2,0.3,320,480,720]; % 
list_x_star = list_x_line*1.13;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(1)*1.01,F1_mean(1),'n.s.','HorizontalAlignment', 'left','FontSize',11,'Color',colors_select(1,:),'Rotation',270);
text(list_x_star(1)*0.95,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',11,'Color',colors_select(1,:),'Rotation',270);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(end),'n.s.','HorizontalAlignment', 'left','FontSize',11,'Color',colors_select(2,:),'Rotation',270);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(end),'n.s.','HorizontalAlignment', 'left','FontSize',11,'Color',colors_select(3,:),'Rotation',270);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(end),'n.s.','HorizontalAlignment', 'left','FontSize',11,'Color',colors_select(4,:),'Rotation',270);
line(list_x_line(5)*[1,1],[F1_mean(5),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1.01,F1_mean(end),'n.s.','HorizontalAlignment', 'left','FontSize',11,'Color',colors_select(5,:),'Rotation',270);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',3); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1.12]);
xlim([10,1000]);
xticks(10.^(1:4));
set(gca,'xticklabel',get(gca,'xtick'));
set(gca,'xticklabelRotation',0);
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['Fig5F - F1 speed spatiotemporal ',data_name,'.emf']);
saveas(gcf,['Fig5F - F1 speed spatiotemporal ',data_name,'.png']);

