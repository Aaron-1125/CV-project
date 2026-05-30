_base_ = "./scrfd_like_r50_fpn_widerface_640_gpu.py"

work_dir = "work_dirs/task3_v2/scrfd_like_r50_fpn_widerface_640_bs1_gpu"
train_dataloader = dict(batch_size=1)
optim_wrapper = dict(optimizer=dict(lr=0.000625))
auto_scale_lr = dict(enable=False, base_batch_size=16)
