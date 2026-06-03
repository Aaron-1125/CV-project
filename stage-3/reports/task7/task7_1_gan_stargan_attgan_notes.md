# Stage3 Task7.1 GAN / StarGAN / AttGAN Notes

## GAN Basics

GAN uses a generator and a discriminator in an adversarial objective. For face
editing, the generator should change the requested semantic attribute while
preserving identity, pose, illumination, and unrelated attributes.

## StarGAN

StarGAN learns multi-domain image translation with one generator. For CelebA,
the target domain is represented by an attribute vector such as
`Black_Hair / Blond_Hair / Brown_Hair / Male / Young`. The same generator can
translate one input face into each target attribute direction.

The official CelebA configuration uses:

- image size: `128`
- crop size: `178`
- attributes: `Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Male`, `Young`
- losses: adversarial loss, domain classification loss, reconstruction loss,
  and gradient penalty

## AttGAN

AttGAN is another attribute editing model that explicitly balances attribute
classification and reconstruction. It is useful as a comparison point: StarGAN
scales multi-domain editing through a unified conditional generator, while
AttGAN focuses on preserving attribute-excluding details through reconstruction
constraints.

## Task7 Evaluation Focus

FID and Inception Score measure distribution-level image quality, but they do
not prove that a requested edit succeeded or that identity was preserved. This
task therefore treats attribute success rate and InsightFace identity cosine as
primary metrics, with FID/IS as auxiliary quality indicators.

