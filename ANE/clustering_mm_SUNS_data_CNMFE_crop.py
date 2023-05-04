# data_ind = 1; dir_SUNS_sub = os.path.join('SUNS_TUnCaT_SF25','4816[1]th5');
# data_ind = 2; dir_SUNS_sub = os.path.join('SUNS_TUnCaT_SF25','4816[1]th4');
# data_ind = 3; dir_SUNS_sub = os.path.join('SUNS_TUnCaT_SF25','4816[1]th4');
# data_ind = 4; dir_SUNS_sub = os.path.join('SUNS_TUnCaT_SF50','4816[1]th4');
import numpy as np
import os
import scipy.io as sio
addpath(genpath('.'))
gcp
##
# name of the videos
list_data_names = np.array(['blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'])
list_ID_part = np.array(['_part11','_part12','_part21','_part22'])
data_name = list_data_names[data_ind]
list_Exp_ID = cellfun(lambda x = None: np.array([data_name,x]),list_ID_part,'UniformOutput',False)
num_Exp = len(list_Exp_ID)
rate_hz = np.array([10,15,7.5,5])

list_avg_radius = np.array([5,6,8,14])
r_bg_ratio = 3
leng = r_bg_ratio * list_avg_radius(data_ind)
th_IoU_split = 0.5
thj_inclass = 0.4
thj = 0.7
meth_baseline = 'median'

meth_sigma = 'quantile-based std'

sub_added = ''
## Load traces and ROIs
# folder of the GT Masks
dir_parent = os.path.join('..','data','data_CNMFE',np.array([data_name,sub_added]))
dir_video = dir_parent
dir_SUNS = os.path.join(dir_parent,dir_SUNS_sub)
dir_masks = os.path.join(dir_SUNS,'output_masks')
# dir_masks = os.path.join(dir_parent, 'GT Masks');
dir_add_new = os.path.join(dir_masks,'add_new_blockwise')
# fs = rate_hz(data_ind);
if not os.path.exist(str(dir_add_new)) :
    mkdir(dir_add_new)

time_weights = np.zeros((num_Exp,1))
for eid in np.arange(1,num_Exp+1).reshape(-1):
    Exp_ID = list_Exp_ID[eid]
    sio.loadmat(os.path.join(dir_masks,np.array(['Output_Masks_',Exp_ID,'.mat'])),'Masks')
    masks = permute(logical(Masks),np.array([3,2,1]))
    #     load(os.path.join(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
#     masks=logical(FinalMasks);
    fname = os.path.join(dir_video,np.array([Exp_ID,'.h5']))
    video_raw = h5read(fname,'/mov')
    tic
    video_sf = homo_filt(video_raw,50)
    mu,sigma = SNR_normalization_video(video_sf,meth_sigma,meth_baseline)
    video_SNR = (video_sf - mu) / sigma
    video_SNR = imgaussfilt(video_SNR)
    Lx,Ly,T = video_SNR.shape
    npatchx = np.ceil(Lx / leng) - 1
    npatchy = np.ceil(Ly / leng) - 1
    num_avg = np.amax(60,np.amin(90,np.ceil(T * 0.01)))
    ## Create memory mapping files for SNR video and masks
    fileName = 'video_SNR.dat'
    if ('mm' is not None):
        clear('mm')
    if os.path.exist(str(fileName)):
        os.delete(fileName)
    fileID = open(fileName,'w')
    video_class = class_(video_SNR)
    fwrite(fileID,video_SNR,video_class)
    fileID.close()
    mm = memmapfile(fileName,'Format',np.array([video_class,np.array([Lx,Ly,T]),'video']),'Repeat',1)
    mm2 = memmapfile(fileName,'Format',np.array([video_class,np.array([Lx * Ly,T]),'video']),'Repeat',1)
    ##
    traces_raw = generate_traces_from_masks_mm(mm2,masks)
    ##
    video = video_SNR
    list_weight,list_weight_trace,list_weight_frame,sum_edges = frame_weight_blockwise_mm(mm,traces_raw,masks,leng)
    ##
    area = np.squeeze(np.sum(np.sum(masks, 1-1), 2-1))
    avg_area = median(area)
    list_added_full,list_added_crop,list_added_images_crop,list_added_frames,list_added_weights,list_locations = deal(cell(npatchx,npatchy))
    ##
    for ix in np.arange(1,npatchx+1).reshape(-1):
        for iy in np.arange(1,npatchy+1).reshape(-1):
            xmin = np.amin(Lx - 2 * leng + 1,(ix - 1) * leng + 1)
            xmax = np.amin(Lx,(ix + 1) * leng)
            ymin = np.amin(Ly - 2 * leng + 1,(iy - 1) * leng + 1)
            ymax = np.amin(Ly,(iy + 1) * leng)
            weight = list_weight[ix,iy]
            image_new_crop,mask_new_crop,mask_new_full,select_frames_class,select_weight_calss = find_missing_blockwise_mm(mm,masks,xmin,xmax,ymin,ymax,weight,num_avg,avg_area,thj,thj_inclass,th_IoU_split)
            list_added_images_crop[ix,iy] = image_new_crop
            list_added_crop[ix,iy] = mask_new_crop
            list_added_full[ix,iy] = mask_new_full
            list_added_frames[ix,iy] = select_frames_class
            list_added_weights[ix,iy] = select_weight_calss
            n_class = image_new_crop.shape[3-1]
            list_locations[ix,iy] = np.multiply(np.array([xmin,xmax,ymin,ymax]),np.ones((n_class,1)))
    ##
    masks_added_full = cat(3,list_added_full[:])
    images_added_crop = cat(3,list_added_images_crop[:])
    masks_added_crop = cat(3,list_added_crop[:])
    patch_locations = cat(1,list_locations[:])
    added_frames = horzcat(list_added_frames[:])
    added_weights = horzcat(list_added_weights[:])
    ## Calculate the neighboring neurons
    N = len(added_frames)
    list_neighbors = cell(1,N)
    masks_sum = np.sum(masks, 3-1)
    for n in np.arange(1,N+1).reshape(-1):
        locations = patch_locations(n,:)
        list_neighbors[n] = masks_sum(np.arange(locations(1),locations(2)+1),np.arange(locations(3),locations(4)+1))
    masks_neighbors_crop = cat(3,list_neighbors[:])
    time_weights = toc
    ## Save Masks
# save(os.path.join(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
#     'list_weight','list_weight_trace','list_weight_frame',...
#     'sum_edges','traces_raw','video','masks');
    sio.save(os.path.join(dir_add_new,np.array([Exp_ID,'_added_auto_blockwise.mat'])),'added_frames','added_weights','masks_added_full','masks_added_crop','images_added_crop','patch_locations','time_weights','masks_neighbors_crop')

##
clear('mm')
os.delete(fileName)
print('Finished this step')