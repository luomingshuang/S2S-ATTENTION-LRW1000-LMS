vid_path = '/home/share/DATASET/LRW1000_Public/images'
trn_txt_path = '/home/share/DATASET/LRW1000_Public/info/trn_1000.txt'
val_txt_path = '/home/share/DATASET/LRW1000_Public/info/val_1000.txt'
tst_txt_path = '/home//share/DATASET/LRW1000_Public/info/tst_1000.txt'
# val_vid_path = '/home/lms/GRID/val'
# anno_path = '/home/lms/GRID/align'
vid_pad = 30
txt_pad = 40
max_epoch = 1000
lr = 1e-7
num_workers = 8
display = 10
test_iter = 1000
img_padding = 75
text_padding = 200
teacher_forcing_ratio = 0.01
# save_dir = 'weights'
save_dir1 = 'weights3'
save_dir2 = 'weights2'
mode = 'backendGRU'
if('finetune' in mode):
    batch_size = 64
else:
    batch_size = 32
# weights1 = 'iteration_54001_epoch_30_cer_0.0616691552036915_wer_0.12696610312053158.pt'
# weights1 = 'lrw1000_34.pt'
weights1 = 'iteration_330000_epoch_17_cer_0.5440287196231915_wer_0.7537551218358709.pt'