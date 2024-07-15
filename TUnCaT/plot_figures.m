figure;
plot(list_crop, mean(mean_corr),'LineWidth',2);
hold on;
plot(list_crop, mean(mean_corr_match),'LineWidth',2);
legend({'All neurons','True neurons'},'Location','SouthEast');
xlabel('Clip percentile');
ylabel('Correlation');
set(gca,'FontSize',12);
title('Correlation of SUNS2+TUnCaT');
saveas(gcf,'Correlation SUNS2+TUnCaT.png')
