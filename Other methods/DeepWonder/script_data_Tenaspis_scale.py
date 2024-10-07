import os
import sys

if True:
	folder = 'D:\\data_TENASPIS\\added_refined_masks\\DeepWonder_scale_full'
	list_Exp_ID = [ 'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', \
					'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M']

	for Exp_ID in list_Exp_ID:
		os.system('python main_RSM_copy.py \
		--RMBG_model_folder arr_202110011541 \
		--RMBG_model_name E_30_Iter_4009 \
		--SEG_model_folder TS3DUnetFFD_20211129-1355 \
		--SEG_model_name seg_30 \
		--test_datasize 20000 \
		--datasets_path "' + folder + '" \
		--datasets_folder ' + Exp_ID + ' \
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
