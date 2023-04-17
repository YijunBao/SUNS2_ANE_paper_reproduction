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
data_name = 'lowBG=5e+03,poisson=1 cv 0307';
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = 1:4;
list_method = {'MIN1PIPE', 'CNMF-E', 'SUNS1', 'SUNS2', 'SUNS1-MF'};
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
colors = color([1,2,4:num_method+1],:);
% colors(3,:)=(1+colors(3,:))/2;
% colors(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,400,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
end
plot(10*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = 1.05+(0:3)*0.07;
list_y_star = list_y_line+0.015;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
% text(fps_mean(3),list_y_star(3)+0.03,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',colors(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(4),list_y_star(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));

list_x_line = 10.^(3.2+(0:3)*0.2);
list_x_star = list_x_line*1.4;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
% line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
% text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

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
box off
% title('Cropped one-photon videos');
two_errorbar_position = get(gca,'Position');
saveas(gcf,['Fig2C - F1 speed ',data_name,'.emf']);
saveas(gcf,['Fig2C - F1 speed ',data_name,'.png']);


%% Figure 3E:
% F1 and speed (log) for all methods for TENASPIS videos
data_name = 'TENASPIS_refined_8par 0407'; % 
% data_name = 'TENASPIS_original_5 0217'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = 1:5;
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
colors = color([1,2,4:num_method+1],:);
% colors(3,:)=(1+colors(3,:))/2;
% colors(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,500],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.95,1.05,1.00,1.10]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));

list_x_line = [5,7.5,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line*1.3;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
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
saveas(gcf,['Fig3E - F1 speed ',data_name,'.emf']);
saveas(gcf,['Fig3E - F1 speed ',data_name,'.png']);


%% Figure 4B: 
% F1 and speed for SUNS1 and SUNS2 with or without SF for TENASPIS videos
data_name = 'TENASPIS_refined_8par 0407'; % 
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
colors = distinguishable_colors(14);
colors = [colors(8,:);color(4,:);colors(7,:);color(5,:)];
% colors = color([1,2,4:num_method+1],:);
% colors(3,:)=(1+colors(3,:))/2;
% colors(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
end
% plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]
% 
list_y_line = 0.9+(0:5)*0.07;
list_y_star = list_y_line+0.015;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(3)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(1),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors(1,:));
line([fps_mean(1),fps_mean(2)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(3),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors(1,:));
line([fps_mean(3),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
line([fps_mean(2),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(1)+0.03,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',colors(2,:));
line([fps_mean(1),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(1,:));
% 
list_x_line = 348-(0:5)*10; % [4,6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line+2;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(2)*[1,1],[F1_mean(1),F1_mean(3)],'color','k','LineWidth',2)
text(list_x_star(2)*1,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(1,:),'Rotation',270);
line(list_x_line(3)*[1,1],[F1_mean(1),F1_mean(2)],'color','k','LineWidth',2)
text(list_x_star(3)*1,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(1,:),'Rotation',270);
line(list_x_line(4)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(4)*1,F1_mean(3),'*','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:),'Rotation',270);
line(list_x_line(5)*[1,1],[F1_mean(2)-0.01,F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(5)*1+4,F1_mean(2),'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',colors(2,:),'Rotation',270);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1,F1_mean(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(1,:),'Rotation',270);

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
saveas(gcf,['Fig4B - F1 speed ',data_name(1:end-8),' noSF.emf']);
saveas(gcf,['Fig4B - F1 speed ',data_name(1:end-8),' noSF.png']);


%% Figure 4D:
% F1 and speed (log) for all methods with ANE for TENASPIS videos
data_name = 'TENASPIS_ANE_4 0217'; % 'TENASPIS_original'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
list_method = cellfun(@(x)[x,'-ANE'],list_method,'UniformOutput',false);
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
colors = color([1,2,4:num_method+1],:);
% colors(3,:)=(1+colors(3,:))/2;
% colors(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.90,0.96,0.93,0.99]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.008;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
line([fps_mean(2),fps_mean(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
line([fps_mean(3),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));

list_x_line = 10.^(1.9-(3:-1:0)*0.15); % [4,6,60,90]; % 
list_x_star = list_x_line*1.25;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(1)*1,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(2)*1,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
text(list_x_star(3)*1,F1_mean(4),'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',colors(3,:),'Rotation',270);
% line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(4)*1,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
% legend(list_method,'Location','SouthEast', 'FontName','Arial','FontSize',14,'NumColumns',1); % 

set(gca, 'XScale', 'log');
% yticks(0.6:0.1:1)
% ylim([0.1,0.8]);
ylim([0,1]);
% xlim([10,10000]);
xlim([0.1,100]);
xticks(10.^(-2:3));
set(gca,'xticklabel',get(gca,'xtick'));
box off
% title('Cropped one-photon videos');
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.emf']);
% saveas(gcf,['F1 speed ',data_name(1:end-1),'7.png']);
saveas(gcf,['Fig4D - F1 speed ',data_name,'.emf']);
saveas(gcf,['Fig4D - F1 speed ',data_name,'.png']);


%% Figure 4G:
% F1 and speed (log) for all methods for TENASPIS videos using initial GT masks
% data_name = 'TENASPIS_refined_7par 0210'; % 
data_name = 'TENASPIS_original_5 0404'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = 1:5;
% Recall = Recall(:,select);
% Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'}; % ,'SUNS2 (no SF)'
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
colors = color([1,2,4:num_method+1],:);
% colors(3,:)=(1+colors(3,:))/2;
% colors(2,:)=(colors(2,:))/2;

% %%
figure('Position',[50,50,450,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
end
plot([20,20],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
xlabel('Speed (Frame/s)');
ylabel('{\itF}_1');
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

list_y_line = [0.95,1.05,1.00,1.10]; % 0.99-(3:-1:0)*0.04;
list_y_star = list_y_line+0.01;
% reorder = [1,2,6,3,4,5];
% list_y_line = list_y_line(reorder);
% list_y_star = list_y_star(reorder);
line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));

list_x_line = [5,7.5,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line*1.3;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

legend([list_method,'Video rate'],'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
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
saveas(gcf,['Fig4G - F1 speed ',data_name,'.emf']);
saveas(gcf,['Fig4G - F1 speed ',data_name,'.png']);


%% Figure S4D:
% F1 and speed (log) for DeepWonder, SUNS2, and SUNS2-ANE for TENASPIS videos
data_name = 'TENASPIS_refined_8par 0407'; % 
% data_name = 'TENASPIS_original_5 0217'; % 
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
colors = [colors(14,:); color([5,6],:)];

% %%
figure('Position',[50,50,400,450],'color','w');
hold on;
% errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
for k = 1:n_colors
    plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
end
for k = 1:n_colors
    errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),fps_std(k),fps_std(k),...
        'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
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
text(fps_mean(2),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(2,:));
line([fps_mean(1),fps_mean(3)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
% line([fps_mean(3),fps_mean(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));

list_x_line = [5,7.5,60,90]*0.3; % 10.^(1.9-(3:-1:0)*0.15);
list_x_star = list_x_line*0.93;
% reorder = [1,5,2,6,4,3];
% list_x_line = list_x_line(reorder);
% list_x_star = list_x_star(reorder);
line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(2)],'color','k','LineWidth',2)
text(list_x_star(1)*1.01,F1_mean(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(2,:),'Rotation',90);
line(list_x_line(2)*[1,1],[F1_mean(1),F1_mean(3)],'color','k','LineWidth',2)
text(list_x_star(2)*1.01,F1_mean(3),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:),'Rotation',90);
% line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
% line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(end)],'color','k','LineWidth',2)
% text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

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
saveas(gcf,['FigS4D - F1 speed ',data_name,' DeepWonder.emf']);
saveas(gcf,['FigS4D - F1 speed ',data_name,' DeepWonder.png']);


%% Figure 5D:
% F1 and speed (log) for all methods for CNMF-E videos
rate_hz = [10,15,15,5]; % frame rate of each video
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_8 0407'];
    load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
    select = 1:5;
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
    colors = color([1,2,4:num_method+1],:);
    % colors(3,:)=(1+colors(3,:))/2;
    % colors(2,:)=(colors(2,:))/2;

    % %%
    % figure('Position',[50,50,450,500],'color','w');
    figure('Position',[50,50,400,450],'color','w');
    hold on;
    % errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
    for k = 1:n_colors
        plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
    end
    for k = 1:n_colors
        errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),min(fps_mean(k)-1,fps_std(k)),fps_std(k),...
            'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
    end
    plot(rate_hz(data_ind)*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
    xlabel('Speed (Frame/s)');
    ylabel('{\itF}_1');
    set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

    % list_y_line = [0.93,0.97,0.95,0.99]; % 0.99-(3:-1:0)*0.04;
    % list_y_star = list_y_line+0.005;
    % % reorder = [1,2,6,3,4,5];
    % % list_y_line = list_y_line(reorder);
    % % list_y_star = list_y_star(reorder);
    % line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
    % line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));
    % 
    % list_x_line = [0.4,0.6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
    % list_x_star = list_x_line*1.3;
    % % reorder = [1,5,2,6,4,3];
    % % list_x_line = list_x_line(reorder);
    % % list_x_star = list_x_star(reorder);
    % line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
    % line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
    % line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
    % line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

    legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',14,'NumColumns',2); % 
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
    box off
    % title('Cropped one-photon videos');
    % saveas(gcf,['F1 speed ',data_name,'.emf']);
    % saveas(gcf,['F1 speed ',data_name,'.png']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),data_name(end-4:end),'.emf']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),data_name(end-4:end),'.png']);
    saveas(gcf,['Fig5D ',num2str(data_ind),'- F1 speed ',data_name(1:end-7),'.emf']);
    saveas(gcf,['Fig5D ',num2str(data_ind),'- F1 speed ',data_name(1:end-7),'.png']);
    % if data_ind == 3
    %     saveas(gcf,['FigS11D - F1 speed ',data_name(1:end-7),'.emf']);
    %     saveas(gcf,['FigS11D - F1 speed ',data_name(1:end-7),'.png']);
    % end
end


%% Figure S4H:
% F1 and speed for DeepWonder, SUNS2, and SUNS2-ANE for CNMF-E videos
rate_hz = [10,15,15,5]; % frame rate of each video
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_8 0407'];
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
    colors = [colors(14,:);color(5,:);color(6,:)];
    % colors(3,:)=(1+colors(3,:))/2;
    % colors(2,:)=(colors(2,:))/2;

    % %%
    figure('Position',[50,50,400,450],'color','w');
    hold on;
    % errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
    for k = 1:n_colors
        plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
    end
    for k = 1:n_colors
        errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),min(fps_mean(k)-1,fps_std(k)),fps_std(k),...
            'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
    end
    plot(rate_hz(data_ind)*[1,1],[0,1],'--k','LineWidth',2); % ,'HandleVisibility','off'
    xlabel('Speed (Frame/s)');
    ylabel('{\itF}_1');
    set(gca,'FontName','Arial','FontSize',18, 'LineWidth',1); %,'Position',[0.12,0.15,0.65,0.8]

    % list_y_line = [0.93,0.97,0.95,0.99]; % 0.99-(3:-1:0)*0.04;
    % list_y_star = list_y_line+0.005;
    % % reorder = [1,2,6,3,4,5];
    % % list_y_line = list_y_line(reorder);
    % % list_y_star = list_y_star(reorder);
    % line([fps_mean(1),fps_mean(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
    % line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));
    % 
    % list_x_line = [0.4,0.6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
    % list_x_star = list_x_line*1.3;
    % % reorder = [1,5,2,6,4,3];
    % % list_x_line = list_x_line(reorder);
    % % list_x_star = list_x_star(reorder);
    % line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
    % line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
    % line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
    % line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

    legend(list_method,'Location','NorthOutside', 'FontName','Arial','FontSize',18,'NumColumns',2); % 
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
        xlim([1,10000]);
    elseif data_ind == 3
        xlim([1,10000]);
    else
        xlim([1,1000]);
    end
    xticks(10.^(0:4));
    set(gca,'xticklabel',get(gca,'xtick'));
    box off
    % title('Cropped one-photon videos');
    % saveas(gcf,['F1 speed ',data_name(1:end-7),' noSF',data_name(end-4:end),'.emf']);
    % saveas(gcf,['F1 speed ',data_name(1:end-7),' noSF',data_name(end-4:end),'.png']);
    saveas(gcf,['FigS4H-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' DeepWonder.emf']);
    saveas(gcf,['FigS4H-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' DeepWonder.png']);
end


%% Figure S5D:
% F1 and speed for SUNS1 and SUNS2 with or without SF for CNMF-E videos
rate_hz = [10,15,15,5]; % frame rate of each video
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_8 0407'];
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
    colors = distinguishable_colors(14);
    colors = [colors(8,:);color(4,:);colors(7,:);color(5,:)];
    % colors(3,:)=(1+colors(3,:))/2;
    % colors(2,:)=(colors(2,:))/2;

    % %%
    figure('Position',[50,50,400,420],'color','w');
    hold on;
    % errorbar(fps_mean,F1_mean,F1_std,F1_std,fps_std,fps_std,'LineStyle','None','LineWidth',2);
    for k = 1:n_colors
        plot(Speed(:,k),F1(:,k),'.','Color',(1+colors(k,:))/2,'HandleVisibility','off'); %
    end
    for k = 1:n_colors
        errorbar(fps_mean(k),F1_mean(k),F1_std(k),F1_std(k),min(fps_mean(k)-1,fps_std(k)),fps_std(k),...
            'LineWidth',2,'LineStyle','None','Color',colors(k,:)); %
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
    % text(fps_mean(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:));
    % line([fps_mean(2),fps_mean(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(2),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(3),list_y_star(2),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(3,:));
    % line([fps_mean(3),fps_mean(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % text(fps_mean(4),list_y_star(1),'**','HorizontalAlignment', 'right','FontSize',14,'Color',colors(4,:));
    % 
    % list_x_line = [0.4,0.6,60,90]; % 10.^(1.9-(3:-1:0)*0.15);
    % list_x_star = list_x_line*1.3;
    % % reorder = [1,5,2,6,4,3];
    % % list_x_line = list_x_line(reorder);
    % % list_x_star = list_x_star(reorder);
    % line(list_x_line(1)*[1,1],[F1_mean(1),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(1)*1.01,F1_mean(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(1,:),'Rotation',90);
    % line(list_x_line(2)*[1,1],[F1_mean(2),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(2)*1.01,F1_mean(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(2,:),'Rotation',90);
    % line(list_x_line(3)*[1,1],[F1_mean(3),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(3)*1.01,F1_mean(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(3,:),'Rotation',90);
    % line(list_x_line(4)*[1,1],[F1_mean(4),F1_mean(5)],'color','k','LineWidth',2)
    % text(list_x_star(4)*1.01,F1_mean(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',colors(4,:),'Rotation',90);

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
    saveas(gcf,['FigS5D-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' noSF.emf']);
    saveas(gcf,['FigS5D-',num2str(data_ind),' - F1 speed ',data_name(1:end-7),' noSF.png']);
end

