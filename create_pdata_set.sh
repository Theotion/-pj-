cd /media/charles/STORAGE/code/20190315MTCNN-Pytorch
/home/charles/anaconda3/envs/DL3.7/bin/python create_dataset.py --class_data_txt_path /media/charles/750GB/WILDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt --class_data_dir /media/charles/750GB/WILDER_FACE/WIDER_train --landmark_data_txt_path /media/charles/750GB/CNN_FacePoint/train/trainImageList.txt --landmark_data_dir /media/charles/750GB/CNN_FacePoint/train --output_path /media/charles/750GB/dataset --create_data_set pnet --p_net_data /media/charles/STORAGE/dataset/P_Net_dataset/P_Net_dataset.txt --r_net_data /media/charles/STORAGE/dataset/R_Net_dataset/R_Net_dataset.txt --o_net_data /media/charles/STORAGE/dataset/O_Net_dataset/O_Net_dataset.txt --save_folder ./MTCNN_weighs
