# Stage3 Task7 StarGAN Attribute Editing Report

## 1. Scope

This report covers only Stage3 Task 7.x: face attribute editing with StarGAN on CelebA. The official StarGAN source is kept outside this Git deliverable at `task/StarGAN`; `stage-3` contains wrappers, configs, reports, summaries, and generated evidence.

## 2. Setup And Fixed Samples

- CelebA prepared: `True`
- fixed sample count: `16`
- selected attrs: `['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']`
- train command return code: `0`
- final checkpoint: `work_dirs/task7/stargan_celeba_128_full/models/200000-G.ckpt` on AutoDL, not committed to Git

Fixed input faces:

![fixed source grid](assets/fixed_samples/fixed_source_grid.jpg)

Target labels are saved before generation in `summaries/fixed_samples_target_labels.json` and `summaries/fixed_samples_target_labels.csv`. Hair-color target labels are validated as mutually exclusive for every generated direction.

## 3. Pretrained Sanity Check

The official CelebA 128 pretrained checkpoint is used as a reference before accepting the self-trained run.

- pretrained checkpoint: `task/StarGAN/stargan_celeba_128/models/200000-G.ckpt` on AutoDL, not committed to Git
- pretrained fixed grid: `assets/pretrained/pretrained_200000_fixed_grid.jpg`

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

- checkpoint: `work_dirs/task7/attribute_classifier/resnet18_5attrs.pt` on AutoDL, not committed to Git
- final exact-match accuracy: `67.53%`
- per-attribute accuracy: `{'Black_Hair': 0.919959979989995, 'Blond_Hair': 0.9564782391195598, 'Brown_Hair': 0.8819409704852427, 'Male': 0.9859929964982491, 'Young': 0.888944472236118}`

Attribute edit success:

| Direction | Samples | Primary success | Strict 5-attr success |
| --- | ---: | ---: | ---: |
| Black_Hair | 512 | 0.00% | 0.00% |
| Blond_Hair | 512 | 1.76% | 1.76% |
| Brown_Hair | 512 | 0.00% | 0.00% |
| Male | 512 | 14.26% | 1.56% |
| Young | 512 | 48.83% | 8.40% |

Identity retention with InsightFace `buffalo_l`:

- valid pairs: `1139` / `2560`
- mean cosine: `0.6810`
- median cosine: `0.6943`
- p10 cosine: `0.5694`
- no generated face: `1421`

FID/IS auxiliary metrics:

- available: `False`
- FID: `N/A`
- Inception Score: `N/A +/- N/A`
- unavailable reason: AutoDL timed out while downloading the Inception feature weights required by `torchmetrics`.
- note: FID/IS are auxiliary; acceptance focuses on attribute success and identity retention.

## 6. Failure Case Analysis

The self-trained model completed 200000 iterations and produced all checkpoint-monitoring grids, but the final edit quality is weak: hair-color success is near zero, the `Young` direction is the strongest but still below 50% primary success, and InsightFace failed to detect a face in many generated images. The visual grids show strong color cast and checker/grid artifacts, so this run is best treated as a completed training-and-evaluation reproduction with a failed-quality conclusion rather than a production-quality effect model.

- Attribute miss: `028136.jpg` direction `Black_Hair`, target `[1, 0, 0, 1, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `025600.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `120849.jpg` direction `Black_Hair`, target `[1, 0, 0, 1, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `063578.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `083306.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 0, 0]`.
- Attribute miss: `022603.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `199242.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Attribute miss: `189733.jpg` direction `Black_Hair`, target `[1, 0, 0, 0, 1]`, predicted `[0, 0, 0, 1, 0]`.
- Identity drift: `172396.jpg` direction `Male`, cosine `0.245`.
- Identity drift: `128504.jpg` direction `Young`, cosine `0.296`.
- Identity drift: `127251.jpg` direction `Brown_Hair`, cosine `0.340`.
- Identity drift: `041226.jpg` direction `Male`, cosine `0.267`.

Common failure modes to inspect manually are: target attribute not activated, identity drift after large gender/age edits, hair-color ambiguity, and local artifacts around hairline, glasses, or background.
