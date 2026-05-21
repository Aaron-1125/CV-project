# Stage 1 Task 2.2 Dataset Exploration Summary

## CelebA

- Source: `datasets.load_dataset('eurecom-ds/celeba', split='train+validation+test')`
- Images: `202599`
- Identities: `10177`
- Attributes: `40`
- Sample grid: `reports/assets/dataset/celeba_samples.png`

Top attribute positive rates:

- `No_Beard`: `0.8349`
- `Young`: `0.7736`
- `Attractive`: `0.5125`
- `Mouth_Slightly_Open`: `0.4834`
- `Smiling`: `0.4821`
- `Wearing_Lipstick`: `0.4724`
- `High_Cheekbones`: `0.455`
- `Male`: `0.4168`
- `Heavy_Makeup`: `0.3869`
- `Wavy_Hair`: `0.3196`

## LFW

- Source: `sklearn.datasets.fetch_lfw_people/fetch_lfw_pairs`
- Images: `13233`
- Identities: `5749`
- 10-fold pairs: `6000`
- Pair target counts: `{'Different persons': 3000, 'Same person': 3000}`
- Sample grid: `reports/assets/dataset/lfw_samples.png`
- Public detection/landmark test images:
  - `reports/assets/inputs/public_lfw/lfw_public_00.jpg`
  - `reports/assets/inputs/public_lfw/lfw_public_01.jpg`
  - `reports/assets/inputs/public_lfw/lfw_public_02.jpg`
  - `reports/assets/inputs/public_lfw/lfw_public_03.jpg`
