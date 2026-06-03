# CV Project Stage 3

阶段三当前只交付任务 7.x：StarGAN 人脸属性编辑。官方 StarGAN 源码放在
`/Users/aaron/Documents/字节实习/task/StarGAN`，`stage-3` 只保存 wrapper、
配置、报告、轻量 summary 和结果图。

## 目录

- `code/task7/`：CelebA 准备、pretrained sanity check、训练/测试、监控采样、评估、报告生成脚本。
- `configs/task7_stargan/`：本地 smoke 和 AutoDL A800 full 配置。
- `reports/task7/`：Task7 报告、结果图、summary JSON。
- `data/`、`work_dirs/`、`checkpoints/`：本地/云端数据和训练输出，不提交 Git。

## AutoDL A800 流程

```bash
export STARGAN_REPO=/root/autodl-tmp/task/StarGAN
export CELEBA_ROOT=/root/autodl-tmp/<celebA_public_dataset_path>

cd /root/autodl-tmp/task
git clone https://github.com/yunjey/StarGAN.git StarGAN
git -C StarGAN checkout 94dd002e93a2863d9b987a937b85925b80f7a19f

cd /root/autodl-tmp/CV-project/stage-3
pip install -r requirements-task7.txt

python code/task7/stage3_task7_prepare_celeba.py --celeba-root "$CELEBA_ROOT" --report-dir reports/task7 --force
python code/task7/stage3_task7_pretrained_sanity.py --config configs/task7_stargan/a800_full.py
python code/task7/stage3_task7_run_stargan.py train --config configs/task7_stargan/a800_full.py
python code/task7/stage3_task7_monitor_samples.py --config configs/task7_stargan/a800_full.py --all-checkpoints
python code/task7/stage3_task7_run_stargan.py test --config configs/task7_stargan/a800_full.py --test-iters 200000
python code/task7/stage3_task7_evaluate.py --config configs/task7_stargan/a800_full.py --test-iters 200000
python code/task7/stage3_task7_write_report.py --config configs/task7_stargan/a800_full.py
```

If Dropbox cannot download the official pretrained checkpoint, upload the
official `celeba-128x128-5attrs.zip` manually and run:

```bash
python code/task7/stage3_task7_pretrained_sanity.py \
  --config configs/task7_stargan/a800_full.py \
  --pretrained-zip /path/to/celeba-128x128-5attrs.zip
```

Resume training:

```bash
python code/task7/stage3_task7_run_stargan.py train \
  --config configs/task7_stargan/a800_full.py \
  --resume-iters 100000
```

## Local Smoke

```bash
python code/task7/stage3_task7_make_tiny_celeba.py --out-dir tmp/tiny_celeba --count 2048
python code/task7/stage3_task7_prepare_celeba.py --config configs/task7_stargan/smoke.py --celeba-root tmp/tiny_celeba --force
python code/task7/stage3_task7_run_stargan.py train --config configs/task7_stargan/smoke.py
python code/task7/stage3_task7_monitor_samples.py --config configs/task7_stargan/smoke.py --all-checkpoints --device cpu
python code/task7/stage3_task7_run_stargan.py test --config configs/task7_stargan/smoke.py --test-iters 2 --device cpu
python code/task7/stage3_task7_evaluate.py --config configs/task7_stargan/smoke.py --test-iters 2 --device cpu --eval-count 4 --skip-identity --skip-fid-is
python code/task7/stage3_task7_write_report.py --config configs/task7_stargan/smoke.py
```
