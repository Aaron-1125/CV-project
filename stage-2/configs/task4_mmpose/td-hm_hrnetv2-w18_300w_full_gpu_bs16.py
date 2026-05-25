_base_ = "./td-hm_hrnetv2-w18_300w_full_gpu.py"

optim_wrapper = dict(optimizer=dict(type="Adam", lr=6.25e-5))

train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=16, num_workers=2)
test_dataloader = val_dataloader

work_dir = "work_dirs/task4/hrnetv2_w18_300w_full_bs16"
