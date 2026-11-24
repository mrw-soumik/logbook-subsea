# Licences: what this project uses, and what to settle before any paid use

For anyone who wants to use this work, and whoever advises them on intellectual property. It summarises published terms as of 2026-11-23
and is not legal advice.

## Summary
- **Code, data and the description model are fine for commercial use with credit.** SubPipe is CC BY 4.0, Qwen3-VL is
  Apache-2.0, and every software dependency is under a permissive licence.
- **The camera and sonar models need a decision.** They start from torchvision weights trained on ImageNet (and COCO
  for sonar). ImageNet's terms allow only non-commercial research and education, and torchvision says each user must
  check whether pretrained weights suit their use. Whether a fine-tuned model inherits those terms is the open
  question for a lawyer. **It can be sidestepped:** trained without those weights, both models lose little
  accuracy (tested 2026-11-23, below).
- **The code has no open-source licence yet.** It is public to read, with all rights reserved; MIT or Apache-2.0
  can be added later.

## Data
| Item | Terms | What it means here |
|---|---|---|
| SubPipe (Oceanscan-MST), all chunks | CC BY 4.0 ([Zenodo record](https://zenodo.org/records/12666132)) | Commercial use allowed with credit. The credit line is in the README, data card, video and draft report |
| Hand marks and ratings in `labels/` | Our own work | Ours to license |

## Starting weights for the models
| Model | Starts from | Terms of the training data | Status |
|---|---|---|---|
| M1 camera: DeepLabV3, MobileNetV3-Large encoder | torchvision ImageNet weights | ImageNet: "only for non-commercial research and educational purposes" ([access agreement](https://image-net.org/accessagreement)) | **Open** for paid use |
| M2 and M6 sonar: Faster R-CNN, MobileNetV3-Large FPN | torchvision COCO weights, with an ImageNet backbone | COCO annotations CC BY 4.0; images under Flickr's terms ([MSCOCO card](https://huggingface.co/datasets/shunk031/MSCOCO/raw/main/README.md), [Papers with Code](https://PapersWithCode.com/datasets/license)); plus ImageNet as above | **Open** for paid use |
| M4 descriptions: Qwen3-VL-8B-Instruct, 4-bit MLX version | Released weights, used as is | Apache-2.0 (model card tag, checked on Hugging Face) | Fine; keep the licence notice |

Torchvision's own documentation says pretrained models "may have their own licenses or terms and conditions derived
from the dataset used for training" and that it is the user's responsibility to check
([PyTorch docs](https://docs.pytorch.org/vision/stable/models)).

**Three ways to close the open item:**
1. **Legal advice** on whether models fine-tuned from these weights carry the dataset terms.
2. **Retrain without them,** starting from random weights. **Tested on 2026-11-23 (`FROM_SCRATCH=1`), the cost is
   small:**
   - camera: mean IoU 0.818 against 0.841, still above the 0.80 target;
   - sonar, high frequency: stricter mAP 0.525 against 0.520;
   - sonar, low frequency: 0.456 against 0.514.

   - sonar trained on three surveys: still meets the 0.90 target on two unseen surveys, at 0.903 (high frequency) and
     0.932 (low frequency), against 0.950 and 0.956 with COCO weights.

   See [EVALUATION.md, M6](EVALUATION.md#without-imagenet-or-coco-starting-weights).
3. **Start from weights trained on data cleared for commercial use.**

## Software
Licences as declared in each package's metadata, at the pinned versions.

| Package | Version | Licence |
|---|---|---|
| torch | 2.15.0 | BSD-3-Clause, with bundled parts under Apache-2.0, BSD-2-Clause, BSL-1.0, ISC, MIT and Zlib |
| torchvision | 0.29.2 | BSD |
| numpy | 2.5.5 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| pandas | 3.1.1 | BSD-3-Clause |
| pillow | 13.0.0 | MIT-CMU |
| matplotlib | 3.11.2 | PSF-based matplotlib licence |
| mlx | 0.32.4 | MIT |
| mlx-vlm | 0.8.0 | MIT |
| transformers | 5.24.0 | Apache-2.0 |
| fastapi | 0.165.0 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| pytest | 9.2.0 | MIT |

No AGPL or other copyleft packages are used. Ultralytics YOLO was avoided for that reason.
