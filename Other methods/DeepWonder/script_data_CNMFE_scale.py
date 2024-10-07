import os
import sys

list_name_video = ['blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps']
for name_video in list_name_video:
	ID_part = ['_part11', '_part12', '_part21', '_part22']
	list_Exp_ID = [name_video+x for x in ID_part]
	folder = os.path.join('E:\\data_CNMFE', name_video, 'DeepWonder_scale_full')

	for Exp_ID in list_Exp_ID:
		os.system('python main_RSM_copy.py \
		--RMBG_model_folder arr_202110011541 \
		--RMBG_model_name E_30_Iter_4009 \
		--SEG_model_folder TS3DUnetFFD_20211129-1355 \
		--SEG_model_name seg_30 \
		--test_datasize 20000 \
		--datasets_path "' + folder + '" \
		--datasets_folder "' + Exp_ID + '" \
		--sub_img_s 1500 \
		--sub_img_w ' + sys.argv[1] + ' \
		--GPU 0')
		# --sub_img_w 480 \
		# --sub_img_h 480 \
		# --sub_gap_w 448 \
		# --sub_gap_h 448 \

# os.system('python main_RSM_copy.py \
# --RMBG_model_folder arr_202110011541 \
# --RMBG_model_name E_30_Iter_4009 \
# --SEG_model_folder TS3DUnetFFD_20211129-1355 \
# --SEG_model_name seg_30 \
# --test_datasize 20000 \
# --datasets_path "E:/data_DeepWonder" \
# --datasets_folder widefield \
# --GPU 0')

# os.system('python main_RSM.py \
# --RMBG_model_folder arr_202110011541 \
# --RMBG_model_name E_30_Iter_4009 \
# --SEG_model_folder TS3DUnetFFD_20211129-1355 \
# --SEG_model_name seg_30 \
# --test_datasize 20000 \
# --datasets_path datasets \
# --datasets_folder Simulated_wide_field_data1 \
# --GPU 0')
