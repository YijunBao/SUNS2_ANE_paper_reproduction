% Set the path of the 'GT Masks' folder, which contains the manual labels in 3D arrays.
list_data_names={'noise30'};
data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\simulation_CNMFE',data_name);
dir_Masks = fullfile(path_name,'GT Masks'); % FinalMasks_
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);

%%
list_area = cell(num_Exp,1);
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_Masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
%     [Lx,Ly,ncells]=size(FinalMasks);
    list_area{eid} = squeeze(sum(sum(FinalMasks)));
end
all_area = cell2mat(list_area);
mean_area = mean(all_area)
mean_radius = sqrt(mean_area/pi)