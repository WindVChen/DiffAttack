<div align="center">

<h1><a href="https://arxiv.org/abs/2305.08192">Diffusion Models for Imperceptible and Transferable Adversarial Attack</a></h1>

**[Jianqi Chen](https://windvchen.github.io/), [Hao Chen](https://scholar.google.com.hk/citations?hl=en&user=BEDNoZIAAAAJ&view_op=list_works&sortby=pubdate), [Keyan Chen](https://scholar.google.com.hk/citations?hl=en&user=5RF4ia8AAAAJ), [Yilan Zhang](https://scholar.google.com.hk/citations?user=wZ4M4ecAAAAJ&hl=en&oi=ao), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), and [Zhenwei Shi](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

![](https://komarev.com/ghpvc/?username=windvchenDiffAttack&label=visitors)
![GitHub stars](https://badgen.net/github/stars/windvchen/DiffAttack)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2305.08192-b31b1b.svg)](https://arxiv.org/abs/2305.08192)

</div>

### Share us a :star: if this repo does help

This repository is the official implementation of *DiffAttack*. If you encounter any question, please feel free to contact us. You can create an issue or just send email to me windvchen@gmail.com. Also welcome for any idea exchange and discussion.

## Updates

[**11/30/2023**] Access the latest version, v2, of our paper on [Arxiv](https://arxiv.org/abs/2305.08192v2). 👋👋 In this updated release, we have enriched the content with additional discussions and experiments. Noteworthy additions include comprehensive experiments on diverse datasets (refer to Appendix I), exploration of various model structures (refer to Appendix H), and insightful comparisons with ensemble attacks (refer to Appendix G & K) as well as GAN-based methods (refer to Appendix J). Furthermore, we provide expanded details on the current limitations and propose potential directions for future research on diffusion-based methods (refer to Section 5).

[**09/07/2023**] Besides **ImageNet-Compatible**, the code now also supports generating adversarial attacks on **CUB_200_2011** and **Standford Cars** datasets. 🚀🚀 Please refer to [Requirements](#Requirements) for more details.

[**05/16/2023**] Code is public.

[**05/14/2023**] Paper is publicly accessible on [ArXiv](https://arxiv.org/abs/2305.08192) now.

[**04/30/2023**] Code cleanup done. Waiting to be made public.


## Table of Contents

- [Abstract](#Abstract)
- [Requirements](#Requirements)
- [Crafting Adversarial Examples](#Crafting-Adversarial-Examples)
- [Evaluation](#Evaluation)
  - [Robustness on other normally trained models](#robustness-on-other-normally-trained-models)
  - [Robustness on defensive approaches](#Robustness-on-defensive-approaches)
- [Results](#Results)
- [Citation & Acknowledgments](#Citation-&-Acknowledgments)
- [License](#License)


## Abstract

![DiffAttack's framework](Figs/Framework.png)

Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose **a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models**. Specifically, instead of direct manipulation in pixel space, we craft perturbations in latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an additional recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, ***DiffAttack***, is **the first that introduces diffusion models into adversarial attack field**. Extensive experiments on various model structures (including CNNs, Transformers, MLPs) and defense methods have demonstrated our superiority over other attack methods.

## Requirements

1. Hardware Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 16GB memory

2. Software Requirements
    - Python: 3.8
    - CUDA: 11.3
    - cuDNN: 8.4.1

   To install other requirements:

   ```
   pip install -r requirements.txt
   ```

3. Datasets
   - There have been demo-datasets in [demo](demo), you can directly run the optimization code below to see the results.
   - If you want to test the full `ImageNet-Compatible` dataset, please download the dataset [ImageNet-Compatible](https://drive.google.com/file/d/1sAD1aVLUsgao1X-mu6PwcBL8s68dm5U9/view?usp=sharing) and then change the settings of `--images_root` and `--label_path` in [main.py](main.py)

4. Pre-trained Models
   - We adopt `Stable Diffusion 2.0` as our diffusion model, you can load the pretrained weight by setting `--pretrained_diffusion_path="stabilityai/stable-diffusion-2-base"` in [main.py](main.py).
   - For the pretrained weights of the adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>) in Section 4.2.2 of our paper, you can download them from [here](https://github.com/ylhz/tf_to_pytorch_model) and then place them into the directory `pretrained_models`.

5. (Supplement) Attack **CUB_200_2011** and **Standford Cars** datasets
   - Dataset: Aligned with **ImageNet-Compatible**, we randomly select 1K images from **CUB_200_2011** and **Standford Cars** datasets, respectively. You can download the dataset here [[CUB_200_2011](https://drive.google.com/file/d/1umBxwhRz6PIG6cli40Fc0pAFl2DFu9WQ/view?usp=sharing) | [Standford Cars](https://drive.google.com/file/d/1FiH98QyyM9YQ70PPJD4-CqOBZAIMlWJL/view?usp=sharing)] and then change the settings of `--images_root` and `--label_path` in [main.py](main.py). Note that you should also set `--dataset_name` to `cub_200_2011` or `standford_car` when running the code.
   - Pre-trained Models: You can download models (ResNet50, SENet154, and SE-ResNet101) pretrained on CUB_200_2011 and Standford Cars from [Beyond-ImageNet-Attack](https://github.com/Alibaba-AAIG/Beyond-ImageNet-Attack) repository. Then place them into the directory `pretrained_models`.

## Crafting Adversarial Examples

To craft adversarial examples, run this command:

```
python main.py --model_name <surrogate model> --save_dir <save path> --images_root <clean images' path> --label_path <clean images' label.txt>
```
The specific surrogate models we support can be found in `model_selection` function in [other_attacks.py](other_attacks.py). You can also leverage the parameter `--dataset_name` to generate adversarial examples on other datasets, such as `cub_200_2011` and `standford_car`.

The results will be saved in the directory `<save path>`, including adversarial examples, perturbations, original images, and logs.

For some specific images that distort too much, you can consider weaken the inversion strength by setting `--start_step` to a larger value, or leveraging pseudo masks by setting `--is_apply_mask=True`.

## Evaluation

### Robustness on other normally trained models

To evaluate the crafted adversarial examples on other black-box models, run:

```
python main.py --is_test True --save_dir <save path> --images_root <outputs' path> --label_path <clean images' label.txt>
```
The `--save_dir` here denotes the path to save only logs. The `--images_root` here should be set to the path of `--save_dir` in above [Crafting Adversarial Examples](#crafting-adversarial-examples).


### Robustness on defensive approaches

Apart from the adversarially trained models, we also evaluate our attack's power to deceive other defensive approaches as displayed in Section 4.2.2 in our paper, their implementations are as follows:
- Adversarially trained models (Adv-Inc-v3, Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub>): Run the code in [Robustness on other normally trained models](#robustness-on-other-normally-trained-models).
- [HGD](https://github.com/lfz/Guided-Denoise): Change the input size to 224, and then directly run the original code.
- [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense): Since our target size is 224, we reset the image scale augmentation proportionally (232~248). Then run the original code.
- [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): Since its ensembled models failed to process inputs with 224 size, we run its original code that resized the inputs to 299 size.
- [RS](https://github.com/locuslab/smoothing): Change the input size to 224 and set sigma=0.25, skip=1, max=-1, N0=100, N=100, alpha=0.001, then run the original code.
- [NRP](https://github.com/Muzammal-Naseer/NRP): Change the input size to 224 and set purifier=NRP, dynamic=True, then run the original code.
- [DiffPure](https://github.com/NVlabs/DiffPure): Modify the original codes to evaluate the existing adversarial examples, not crafted examples again.

## Results

![Visual comparisons](Figs/Visual.png#pic_center)
![Robustness on normally trained models](Figs/Normal.png#pic_center)
![Robustness on defensive approaches](Figs/Defensive.png#pic_center)


## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```
@article{chen2023diffusion,
  title={Diffusion Models for Imperceptible and Transferable Adversarial Attack},
  author={Chen, Jianqi and Chen, Hao and Chen, Keyan and Zhang, Yilan and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2305.08192},
  year={2023}
}
```

Also thanks for the open source code of [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt). Some of our codes are based on them.

## License
This project is licensed under the Apache-2.0 license. See [LICENSE](LICENSE) for details.