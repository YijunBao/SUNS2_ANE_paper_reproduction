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

%% Figure 2 - figure supplement 3: 
% Plot F1 of all methods processing simulated videos with different background and noise levels
load('F1_simu_lowpassBG 0311.mat','list_Recall','list_Precision','list_F1','list_params','list_method');
% list_method = {'MIN1PIPE', 'CNMF-E', 'SUNS1', 'SUNS2', 'SUNS1-MF'};
list_params = { 'Low b.g.,Low noise'; 'High b.g.,Low noise'; ...
                'Low b.g.,Medium noise'; 'High b.g.,Medium noise'; ...
                'Low b.g.,High noise'; 'High b.g.,High noise'};
                
array_F1 = cell2mat(reshape(list_F1,1,1,length(list_F1)));
array_F1 = array_F1(:,1:4,:);
data = squeeze(array_F1(end-1,:,:))';
err = squeeze(array_F1(end,:,:))'; %/sqrt(9)

figure('Position',[100,100,900,600],'Color','w');
b=bar(data);       
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
% b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(3,:);
ylim([0.0,1.0]);
hold on
ylabel('{\itF}_1')
% xticklabels(list_params);
xticklabels(cellfun(@(x) replace(x,',','\newline'), list_params,'UniformOutput',false));
legend(list_method,'Location','NorthOutside','NumColumns',5,'FontSize',12); % 

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,squeeze(array_F1(1:10,i,:)),'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None','HandleVisibility','off');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1,'HandleVisibility','off');
end
set(gca,'FontName','Arial','FontSize',12, 'LineWidth',1);
% box off;
% saveas(gcf,'F1_simu_CNMFEBG 0813.png')
% saveas(gcf,'F1_simu_lowpassBG 1221.png')
% saveas(gcf,'Fig2S-3 - F1_simu_lowpassBG_all.svg')Fig3
saveas(gcf,'Fig2S-3 - F1_simu_lowpassBG_all.emf')
saveas(gcf,'Fig2S-3 - F1_simu_lowpassBG_all.png')

%% Figure 3 - figure supplement 2A: 
% Comparing Recall, Precision, and F1 of SUNS1 and SUNS2 with or without SF for TENASPIS dataset.
data_name = 'TENASPIS_refined_7par 0210'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE','SUNS1 (no SF)','SUNS2 (no SF)'};
select = [6,3,7,4];
Recall = Recall(:,select);
Precision = Precision(:,select);
F1 = F1(:,select);
Speed = Speed(:,select);
list_method = list_method(:,select);

data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,450,450],'Color','w');
b=bar(data);       
ylim([0.0,1.2]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Score')
% xlabel('Preprocessing methods')
xticklabels({'Recall','Precision','{\itF}_1'});
% title('Accuracy between different methods');
colors = distinguishable_colors(14);
b(1).FaceColor  = colors(8,:);
b(3).FaceColor  = colors(7,:);
b(2).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
% b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(7,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.95+(0:4)*0.06;
list_y_star = list_y_line+0.01;
line([xpoints(1),xpoints(3)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(1),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints(2),xpoints(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(2),list_y_star(3)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',b(2).FaceColor);
line([xpoints(1),xpoints(2)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints(3),xpoints(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints(3),list_y_star(1),'*','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
line([xpoints(1),xpoints(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(xpoints(1),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig3S-2A - Scores ',data_name(1:end-8),' noSF.emf']);
saveas(gcf,['Fig3S-2A - Scores ',data_name(1:end-8),' noSF.png']);

%% Figure 3C: 
% Comparing Recall, Precision, and F1 of different segmentation methods on TENASPIS dataset.
data_name = 'TENASPIS_refined_7par 0210'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = 1:5;
Recall = Recall(:,select);
Precision = Precision(:,select);
F1 = F1(:,select);
% Speed = Speed(:,select);
list_method = list_method(:,select);
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'};  % ,'SUNS2 (no SF)'
data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,450,500],'Color','w');
b=bar(data);       
% ylim([0.0,1.2]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Score')
% xlabel('Preprocessing methods')
xticklabels({'Recall','Precision','{\itF}_1'});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(7,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.95+(0:4)*0.05;
list_y_star = list_y_line+0.01;
line([xpoints(1),xpoints(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints(2),xpoints(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
line([xpoints(3),xpoints(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
line([xpoints(4),xpoints(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);
% line([xpoints(6),xpoints(4)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
% text(xpoints(5),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(6).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig3C - Scores ',data_name,'.emf']);
saveas(gcf,['Fig3C - Scores ',data_name,'.png']);

%% Figure 3 - figure supplement 4A: 
% Comparing Recall, Precision, and F1 of different segmentation methods on TENASPIS dataset using original GT masks.
data_name = 'TENASPIS_original_5 0217'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = 1:5;
Recall = Recall(:,select);
Precision = Precision(:,select);
F1 = F1(:,select);
% Speed = Speed(:,select);
list_method = list_method(:,select);
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'};  % ,'SUNS2 (no SF)'
data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,450,500],'Color','w');
b=bar(data);       
% ylim([0.0,1.2]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Score')
% xlabel('Preprocessing methods')
xticklabels({'Recall','Precision','{\itF}_1'});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(7,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.95+(0:4)*0.05;
list_y_star = list_y_line+0.01;
line([xpoints(1),xpoints(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints(2),xpoints(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
line([xpoints(3),xpoints(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
line([xpoints(4),xpoints(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);
% line([xpoints(6),xpoints(4)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
% text(xpoints(5),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(6).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig3S-4A - Scores ',data_name,'.emf']);
saveas(gcf,['Fig3S-4A - Scores ',data_name,'.png']);

%% Figure 3 - figure supplement 3A:
% Comparing Recall, Precision, and F1 of different segmentation methods with ANE for TENASPIS dataset.
data_name = 'TENASPIS_ANE_4 0217'; % 
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
% list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'};  % ,'SUNS2 (no SF)'
list_method = cellfun(@(x)[x,'-ANE'],list_method,'UniformOutput',false);
data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,450,450],'Color','w');
b=bar(data);       
ylim([0.0,1.1]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Score')
% xlabel('Preprocessing methods')
xticklabels({'Recall','Precision','{\itF}_1'});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
% b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(7,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.95+(0:4)*0.05;
list_y_star = list_y_line+0.01;
line([xpoints(1),xpoints(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints(2),xpoints(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
line([xpoints(3),xpoints(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(3),list_y_star(3)+0.03,'n.s.','HorizontalAlignment', 'left','FontSize',12,'Color',b(3).FaceColor);
% line([xpoints(4),xpoints(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);
% line([xpoints(6),xpoints(4)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
% text(xpoints(5),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(6).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig3S-3A - Scores ',data_name,'.emf']);
saveas(gcf,['Fig3S-3A - Scores ',data_name,'.png']);


%% Figure 3 - figure supplement 4C: 
% Comparing Recall, Precision, and F1 of different segmentation methods for TENASPIS dataset using different GT sets.
data_name_refined = 'TENASPIS_refined_7par 0210'; % 
refined = load(['F1 speed ',data_name_refined,'.mat'],'Recall','Precision','F1','Speed','list_method');
data_name_original = 'TENASPIS_original_5 0217'; % 
original = load(['F1 speed ',data_name_original,'.mat'],'Recall','Precision','F1','Speed','list_method');
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'};  % ,'SUNS2 (no SF)'

Precision = original.F1;
F1 = refined.F1(:,1:5);

data = [Precision(end-1,:); F1(end-1,:)];
err = [Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,320,450],'Color','w');
b=bar(data);       
ylim([0.0,1.2]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('{\itF}_1')
% xlabel('Preprocessing methods')
xticklabels({'Initial\newlinelabels','Refined\newlinelabels'});
% xticklabels({["Initial"+newline+"labels"],['Refined',newline,'labels']});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(7,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints1=1 - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
xpoints2=2 - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.9+(0:4)*0.07;
list_y_star = list_y_line+0.01;
line([xpoints1(1),xpoints2(1)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints1(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints1(2),xpoints2(2)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints1(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
line([xpoints1(3),xpoints2(3)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints1(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
line([xpoints1(4),xpoints2(4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(xpoints1(4),list_y_star(4),'*','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);
line([xpoints1(5),xpoints2(5)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
text(xpoints1(5),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(5).FaceColor);
% text(xpoints2(5),list_y_star(5)+0.035,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(5).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig3S-4C - F1 TENASPIS original vs refined.emf']);
saveas(gcf,['Fig3S-4C - F1 TENASPIS original vs refined.png']);

%% Figure 3 - figure supplement 3C: 
% Comparing Recall, Precision, and F1 of different segmentation methods with or without ANE on TENASPIS dataset.
data_name_init = 'TENASPIS_refined_7par 0210'; % 'TENASPIS_original'; % 
init = load(['F1 speed ',data_name_init,'.mat'],'Recall','Precision','F1','Speed','list_method');
data_name_ANE = 'TENASPIS_ANE_4 0217'; % 'TENASPIS_ANE(accept_all)_4 0217'; % 
ANE = load(['F1 speed ',data_name_ANE,'.mat'],'Recall','Precision','F1','Speed','list_method');
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2'};  % ,'SUNS2-ANE','SUNS2 (no SF)'

Precision = init.F1(:,1:4);
F1 = ANE.F1(:,1:4);

data = [Precision(end-1,:); F1(end-1,:)];
err = [Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,320,450],'Color','w');
b=bar(data);       
ylim([0.0,1.1]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('{\itF}_1')
% xlabel('Preprocessing methods')
xticklabels({'Without\newlineANE','With\newlineANE'});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
% b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(7,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints1=1 - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
xpoints2=2 - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 0.88+(0:4)*0.05;
list_y_star = list_y_line+0.01;
line([xpoints1(1),xpoints2(1)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
text(xpoints1(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints1(2),xpoints2(2)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints1(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
line([xpoints1(3),xpoints2(3)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints1(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
line([xpoints1(4),xpoints2(4)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
text(xpoints1(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);
% line([xpoints1(5),xpoints2(5)],list_y_line(5)*[1,1],'color','k','LineWidth',2)
% text(xpoints1(5),list_y_star(5),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(5).FaceColor);
% text(xpoints2(5),list_y_star(5)+0.035,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(5).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig3S-3C - F1 TENASPIS ANE all.emf']);
saveas(gcf,['Fig3S-3C - F1 TENASPIS ANE all.png']);


%% Figure 2B:
% Comparing Recall, Precision, and F1 of different segmentation methods for simulated dataset.
data_name = 'lowBG=5e+03,poisson=1 cv 0307';
load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
select = 1:4;
list_method = {'MIN1PIPE', 'CNMF-E', 'SUNS1', 'SUNS2', 'SUNS1-MF'};
list_method = list_method(select);
Recall = Recall(:,select);
Precision = Precision(:,select);
F1 = F1(:,select);
% Speed = Speed(:,select);

data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,400,450],'Color','w');
b=bar(data);       
ylim([0.0,1.2]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Score')
% xlabel('Preprocessing methods')
xticklabels({'Recall','Precision','{\itF}_1'});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
% b(5).FaceColor  = color(6,:);
% b(6).FaceColor  = color(3,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
list_y_line = 1.05+(0:4)*0.05;
list_y_star = list_y_line+0.01;
line([xpoints(1),xpoints(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
line([xpoints(2),xpoints(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
line([xpoints(3),xpoints(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
text(xpoints(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
% line([xpoints(4),xpoints(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig2B - Scores ',data_name,'.emf']);
saveas(gcf,['Fig2B - Scores ',data_name,'.png']);

%% Figure 2 - figure supplement 2: 
% Comparing Recall, Precision, and F1 of SUNS2 and SUNS2-ANE for simulated dataset.
% load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
load('F1_simu_lowpassBG 0311.mat','list_Recall','list_Precision','list_F1',...
    'list_params','list_method'); % 'list_Speed',
params_select = 2;
data_name = list_params{params_select};
select = [4,4];
list_method = {'SUNS2', 'SUNS2-ANE'};
% list_method = {'MIN1PIPE', 'CNMF-E', 'SUNS1', 'SUNS2', 'SUNS1-MF'};
% list_method = list_method(select);
Recall = list_Recall{params_select}(:,select);
Precision = list_Precision{params_select}(:,select);
F1 = list_F1{params_select}(:,select);
data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
figure('Position',[100,100,400,400],'Color','w');
b=bar(data);       
ylim([0.98,1.0]);
hold on
% errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
ylabel('Score')
% xlabel('Preprocessing methods')
xticklabels({'Recall','Precision','{\itF}_1'});
% title('Accuracy between different methods');
% colors = distinguishable_colors(14);
% b(2).FaceColor  = colors(7,:);
% b(3).FaceColor  = color(4,:);
b(1).FaceColor  = color(5,:);
b(2).FaceColor  = color(6,:);
% b(6).FaceColor  = color(3,:);

numgroups = size(data,1); 
numbars = size(data,2); 
groupwidth = min(0.8, numbars/(numbars+1.5));
for i = 1:numbars
      % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
      x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
      plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
          'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
      errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
end
set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

% xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
% list_y_line = 1.05+(0:4)*0.05;
% list_y_star = list_y_line+0.01;
% line([xpoints(1),xpoints(end)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
% % text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
% text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
% line([xpoints(2),xpoints(end)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
% text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
% line([xpoints(3),xpoints(end)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
% text(xpoints(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
% % line([xpoints(4),xpoints(end)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
% % text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);

legend(list_method,'Location','northoutside','FontName','Arial','FontSize',14,'NumColumns',2);
box off
% set(gca,'Position',two_errorbar_position);
saveas(gcf,['Fig2S-2 - Scores ',data_name,' ANE.emf']);
saveas(gcf,['Fig2S-2 - Scores ',data_name,' ANE.png']);

%% Figure 4C:
% Comparing Recall, Precision, and F1 of different segmentation methods on CNMF-E dataset.
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_7 0225'];
    load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
    select = 1:5;
    Recall = Recall(:,select);
    Precision = Precision(:,select);
    F1 = F1(:,select);
    % Speed = Speed(:,select);
    list_method = list_method(:,select);
    % list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'};
    data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
    err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
    figure('Position',[100,100,450,500],'Color','w');
    b=bar(data);       
    ylim([0.0,1.0]);
    hold on
    % errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
    ylabel('Score')
    % xlabel('Preprocessing methods')
    xticklabels({'Recall','Precision','{\itF}_1'});
    % title('Accuracy between different methods');
    % colors = distinguishable_colors(14);
    % b(2).FaceColor  = colors(7,:);
    b(3).FaceColor  = color(4,:);
    b(4).FaceColor  = color(5,:);
    b(5).FaceColor  = color(6,:);
    % b(6).FaceColor  = color(3,:);

    numgroups = size(data,1); 
    numbars = size(data,2); 
    groupwidth = min(0.8, numbars/(numbars+1.5));
    for i = 1:numbars
          % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
          x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
          plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
              'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
          errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
    end
    set(gca,'FontName','Arial','FontSize',18, 'LineWidth',1);

    % xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
    % list_y_line = 0.95+(0:4)*0.05;
    % list_y_star = list_y_line+0.01;
    % line([xpoints(1),xpoints(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % % text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
    % text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
    % line([xpoints(2),xpoints(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
    % line([xpoints(3),xpoints(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(xpoints(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
    % line([xpoints(4),xpoints(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);

    legend(list_method,'Location','NorthOutside','FontName','Arial','FontSize',18,'NumColumns',2);
    % legend(list_method,'Location','NorthOutside','FontName','Arial','FontSize',18,'NumColumns',size(F1,2));
    box off
    % set(gca,'Position',two_errorbar_position);
    % saveas(gcf,['Scores ',data_name,'.emf']);
    % saveas(gcf,['Scores ',data_name,'.png']);
    % saveas(gcf,['Scores ',data_name(1:end-7),data_name(end-4:end),'.emf']);
    % saveas(gcf,['Scores ',data_name(1:end-7),data_name(end-4:end),'.png']);
    saveas(gcf,['Fig4C-',num2str(data_ind),' - Scores ',data_name(1:end-7),' 0315.emf']);
    saveas(gcf,['Fig4C-',num2str(data_ind),' - Scores ',data_name(1:end-7),' 0315.png']);
    if data_ind == 3
        saveas(gcf,['Fig4S-3C - Scores ',data_name(1:end-7),' 0315.emf']);
        saveas(gcf,['Fig4S-3C - Scores ',data_name(1:end-7),' 0315.png']);
    end
end

%% Figure 4 - figure supplement 2A:
% Comparing Recall, Precision, and F1 of SUNS1 and SUNS2 with or without SF on the CNMF-E dataset.
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
for data_ind = 1:4
    Exp_ID = list_data_names{data_ind};
    data_name = [Exp_ID,'_refined_7 0225'];
    load(['F1 speed ',data_name,'.mat'],'Recall','Precision','F1','Speed','list_method');
    list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE','SUNS1 (no SF)','SUNS2 (no SF)'};
    select = [6,3,7,4];
    Recall = Recall(:,select);
    Precision = Precision(:,select);
    F1 = F1(:,select);
    Speed = Speed(:,select);
    list_method = list_method(:,select);

    data = [Recall(end-1,:); Precision(end-1,:); F1(end-1,:)];
    err = [Recall(end,:); Precision(end,:); F1(end,:)]; %/sqrt(9)
    figure('Position',[100,100,400,420],'Color','w');
    b=bar(data);       
    ylim([0.0,1.0]);
    hold on
    % errorbar(x,data,err,err, 'Color','k','LineStyle','None','LineWidth',2);    
    ylabel('Score')
    % xlabel('Preprocessing methods')
    xticklabels({'Recall','Precision','{\itF}_1'});
    % title('Accuracy between different methods');
    colors = distinguishable_colors(14);
    b(1).FaceColor  = colors(8,:);
    b(3).FaceColor  = colors(7,:);
    b(2).FaceColor  = color(4,:);
    b(4).FaceColor  = color(5,:);

    numgroups = size(data,1); 
    numbars = size(data,2); 
    groupwidth = min(0.8, numbars/(numbars+1.5));
    for i = 1:numbars
          % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
          x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
          plot(x,[Recall(1:end-2,i),Precision(1:end-2,i),F1(1:end-2,i)],'o',...
              'MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5],'LineStyle','None');
          errorbar(x, data(:,i), err(:,i), 'k', 'linestyle', 'none', 'lineWidth', 1);
    end
    set(gca,'FontName','Arial','FontSize',14, 'LineWidth',1);

    % xpoints=numgroups - groupwidth/2 + (2*(1:numbars)-1) * groupwidth / (2*numbars);
    % list_y_line = 0.95+(0:4)*0.05;
    % list_y_star = list_y_line+0.01;
    % line([xpoints(1),xpoints(5)],list_y_line(1)*[1,1],'color','k','LineWidth',2)
    % % text(xpoints(2),list_y_star(1)+0.01,'n.s.','HorizontalAlignment', 'right','FontSize',12,'Color',b(2).FaceColor);
    % text(xpoints(1),list_y_star(1),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(1).FaceColor);
    % line([xpoints(2),xpoints(5)],list_y_line(2)*[1,1],'color','k','LineWidth',2)
    % text(xpoints(2),list_y_star(2),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(2).FaceColor);
    % line([xpoints(3),xpoints(5)],list_y_line(3)*[1,1],'color','k','LineWidth',2)
    % text(xpoints(3),list_y_star(3),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(3).FaceColor);
    % line([xpoints(4),xpoints(5)],list_y_line(4)*[1,1],'color','k','LineWidth',2)
    % text(xpoints(4),list_y_star(4),'**','HorizontalAlignment', 'left','FontSize',14,'Color',b(4).FaceColor);

    legend(list_method,'Location','NorthOutside','FontName','Arial','FontSize',14,'NumColumns',2);
    box off
    % set(gca,'Position',two_errorbar_position);
    saveas(gcf,['Fig4S-2A-',num2str(data_ind),' - Scores ',data_name(1:end-7),' noSF',data_name(end-4:end),'.emf']);
    saveas(gcf,['Fig4S-2A-',num2str(data_ind),' - Scores ',data_name(1:end-7),' noSF',data_name(end-4:end),'.png']);
end

%% Figure 3D:
% Comparing train speed on the TENASPIS dataset
figure('Position',[100,100,700,200],'Color','w');
load('train_time_TENASPIS.mat','train_time','list_method');
list_method = {'MIN1PIPE','CNMF-E','SUNS1','SUNS2','SUNS2-ANE'};
train_time = train_time/60;
% subplot(1,3,1);
% set(gca,'Position',[0.07,0.1,0.25,0.65])
% data1=[3.91	3.3	27.1	52.5	33.5	7.9];
% data10=[13.01	12.28	181.9	397.2	335	70.9];
data1 = train_time(1,:);
data8 = 0*train_time(2,:);
% select = [2:4,6];
% data1=data1(select);
% data8=data8(select);
data=[data1;data8/8];

b=bar(data);
b(3).FaceColor  = color(4,:);
b(4).FaceColor  = color(5,:);
b(5).FaceColor  = color(6,:);
ylim([1,1000]);
yticks(10.^(0:3));
set(gca,'yticklabel',get(gca,'ytick'));
ylabel('Training time (hr)')
box('off');
set(gca,'XTickLabel',{'1 round','8-round average'})
legend(list_method,'Location','EastOutside');
set(gca,'FontSize',14,'LineWidth',1);
set(gca, 'YScale', 'log');
% title('ABO cross validation')

% numgroups = size(data,1); 
% numbars = size(data,2); 
% groupwidth = min(0.8, numbars/(numbars+1.5));
% hold on;
% for i=1:numbars
%     x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
%     errorbar(x(1), data(1,i), err(1,i), 'k', 'linestyle', 'none', 'lineWidth', 1,'HandleVisibility','off');
% end

saveas(gcf,'Fig3D - Training time TENASPIS 0315 single.emf');
saveas(gcf,'Fig3D - Training time TENASPIS 0315 single.png');

