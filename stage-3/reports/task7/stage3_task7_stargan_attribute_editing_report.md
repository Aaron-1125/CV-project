# Stage3 Task7 StarGAN Attribute Editing Report

## 1. Scope

This report covers only Stage3 Task 7.x: face attribute editing with StarGAN on CelebA. The official StarGAN source is kept outside this Git deliverable at `/Users/aaron/Documents/字节实习/task/StarGAN`; `stage-3` contains wrappers, configs, reports, summaries, and generated evidence.

## 2. Setup And Fixed Samples

- CelebA prepared: `True`
- fixed sample count: `16`
- selected attrs: `['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']`
- train command return code: `0`
- final checkpoint: `/root/autodl-tmp/CV-project/stage-3/work_dirs/task7/stargan_celeba_128_full/models/200000-G.ckpt`

Fixed input faces:

![fixed source grid](assets/fixed_samples/fixed_source_grid.jpg)

Target labels are saved before generation in `/root/autodl-tmp/CV-project/stage-3/reports/task7/summaries/fixed_samples_target_labels.json` and `/root/autodl-tmp/CV-project/stage-3/reports/task7/summaries/fixed_samples_target_labels.csv`. Hair-color target labels are validated as mutually exclusive for every generated direction.

## 3. Pretrained Sanity Check

The official CelebA 128 pretrained checkpoint is used as a reference before accepting the self-trained run.

- pretrained checkpoint: `/root/autodl-tmp/task/StarGAN/stargan_celeba_128/models/200000-G.ckpt`
- pretrained fixed grid: `/root/autodl-tmp/CV-project/stage-3/reports/task7/assets/pretrained/pretrained_200000_fixed_grid.jpg`

![official pretrained fixed grid](assets/pretrained/pretrained_200000_fixed_grid.jpg)

## 4. Self-trained Results

Final self-trained fixed grid:

![self trained fixed grid](assets/self_trained/iter_200000/self_trained_200000_fixed_grid.jpg)

Intermediate fixed-sample monitoring:

- iter `10000`: ![monitor iter 10000](assets/monitor/iter_10000/monitor_10000_fixed_grid.jpg)
- iter `20000`: ![monitor iter 20000](assets/monitor/iter_20000/monitor_20000_fixed_grid.jpg)
- iter `30000`: ![monitor iter 30000](assets/monitor/iter_30000/monitor_30000_fixed_grid.jpg)
- iter `40000`: ![monitor iter 40000](assets/monitor/iter_40000/monitor_40000_fixed_grid.jpg)
- iter `50000`: ![monitor iter 50000](assets/monitor/iter_50000/monitor_50000_fixed_grid.jpg)
- iter `60000`: ![monitor iter 60000](assets/monitor/iter_60000/monitor_60000_fixed_grid.jpg)
- iter `70000`: ![monitor iter 70000](assets/monitor/iter_70000/monitor_70000_fixed_grid.jpg)
- iter `80000`: ![monitor iter 80000](assets/monitor/iter_80000/monitor_80000_fixed_grid.jpg)
- iter `90000`: ![monitor iter 90000](assets/monitor/iter_90000/monitor_90000_fixed_grid.jpg)
- iter `100000`: ![monitor iter 100000](assets/monitor/iter_100000/monitor_100000_fixed_grid.jpg)
- iter `110000`: ![monitor iter 110000](assets/monitor/iter_110000/monitor_110000_fixed_grid.jpg)
- iter `120000`: ![monitor iter 120000](assets/monitor/iter_120000/monitor_120000_fixed_grid.jpg)
- iter `130000`: ![monitor iter 130000](assets/monitor/iter_130000/monitor_130000_fixed_grid.jpg)
- iter `140000`: ![monitor iter 140000](assets/monitor/iter_140000/monitor_140000_fixed_grid.jpg)
- iter `150000`: ![monitor iter 150000](assets/monitor/iter_150000/monitor_150000_fixed_grid.jpg)
- iter `160000`: ![monitor iter 160000](assets/monitor/iter_160000/monitor_160000_fixed_grid.jpg)
- iter `170000`: ![monitor iter 170000](assets/monitor/iter_170000/monitor_170000_fixed_grid.jpg)
- iter `180000`: ![monitor iter 180000](assets/monitor/iter_180000/monitor_180000_fixed_grid.jpg)
- iter `190000`: ![monitor iter 190000](assets/monitor/iter_190000/monitor_190000_fixed_grid.jpg)
- iter `200000`: ![monitor iter 200000](assets/monitor/iter_200000/monitor_200000_fixed_grid.jpg)

## 5. Quantitative Evaluation

Attribute classifier:

- checkpoint: `/root/autodl-tmp/CV-project/stage-3/work_dirs/task7/attribute_classifier/resnet18_5attrs.pt`
- final exact accuracy: `67.53%`
- per-attribute accuracy: `{'Black_Hair': 0.919959979989995, 'Blond_Hair': 0.9564782391195598, 'Brown_Hair': 0.8819409704852427, 'Male': 0.9859929964982491, 'Young': 0.888944472236118}`

Attribute edit success:

| Direction | Samples | Primary success | Strict 5-attr success |
| --- | ---: | ---: | ---: |
| Black_Hair | 512 | 88.87% | 88.48% |
| Blond_Hair | 512 | 83.79% | 81.84% |
| Brown_Hair | 512 | 85.74% | 85.55% |
| Male | 512 | 96.88% | 94.73% |
| Young | 512 | 93.16% | 92.38% |

Identity retention with InsightFace `buffalo_l`:

- valid pairs: `2434` / `2560`
- mean cosine: `0.6215`
- median cosine: `0.6295`
- p10 cosine: `0.4699`
- no generated face: `124`

FID/IS auxiliary metrics:

- available: `False`
- FID: `N/A`
- Inception Score: `N/A +/- N/A`
- note: FID/IS are auxiliary; acceptance focuses on attribute success and identity retention.

## 6. Failure Case Analysis

- Attribute miss: `180182.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 0, 1]`.
- Attribute miss: `193652.jpg` direction `Black_Hair`, target `[1, 0, 0, 1, 0]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `027825.jpg` direction `Black_Hair`, target `[1, 0, 0, 1, 1]`, predicted `[0, 0, 0, 1, 1]`.
- Attribute miss: `016160.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 0, 1]`.
- Attribute miss: `174667.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 0, 1]`.
- Attribute miss: `129818.jpg` direction `Black_Hair`, target `[1, 0, 0, 1, 0]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `201256.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 0, 1]`.
- Attribute miss: `040792.jpg` direction `Black_Hair`, target `[1, 0, 0, 1, 0]`, predicted `[0, 0, 0, 1, 1]`.
- Identity drift: `018394.jpg` direction `Blond_Hair`, cosine `0.209`.
- Identity drift: `005109.jpg` direction `Male`, cosine `0.328`.
- Identity drift: `174667.jpg` direction `Male`, cosine `0.326`.
- Identity drift: `040792.jpg` direction `Male`, cosine `0.331`.
- Identity drift: `005109.jpg` direction `Young`, cosine `0.324`.
- Identity drift: `174667.jpg` direction `Young`, cosine `0.314`.
- Identity drift: `129839.jpg` direction `Male`, cosine `0.327`.
- Identity drift: `182446.jpg` direction `Male`, cosine `0.348`.

Common failure modes to inspect manually are: target attribute not activated, identity drift after large gender/age edits, hair-color ambiguity, and local artifacts around hairline, glasses, or background.
