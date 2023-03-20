% load('eval_TENASPIS_thb history 20221216.mat','history','best_history')
history_unique1 = unique(history_time(:,end));
history_unique = history_time;

for h = 2:size(history_time,1)
    if ~all(any(history_time(1:h-1,:)-history_time(h,:),2))
        history_unique(h,:) = 0;
    end
end
total_time1 = sum(history_unique1)/60;
total_time = sum(history_unique(:,end))/60;
[total_time1,total_time]
[total_time1/1440,total_time/1440]
