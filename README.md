# Energy attack on Dialogue Generation

## Quickstart

### Setup Environment
- python 3.10.6
- pytorch 1.13.0
- Install dependencies
```
pip install -r requirements.txt
```

### Train and evaluate a model on a specific task(s)

- BART on BlendedSkillTalk:
```
python train_seq2seq.py
```
- Or
```
bash train.sh
```
- Or you can directly download our pre-trained [model](https://drive.google.com/drive/folders/1rWexrwHCgCFYiNVk2yFKSI8iV8baWfFt?usp=sharing) 

### Attack a pre-trained model
```
python attack.py
```

### Plan
- prepare models
- refactor ```attack.py```
- implement some attacking methods
- quantitative experiments
  - get preliminary results
  - scale the experiments
  - evaluate ASR, BLEU, Perplex, distance metrics like SSIM, FID, ...
  - ablation study
- qualitative experiments
  - visualize adversarial examples
  - interpret adversarial examples (Grad-CAM, attention, etc.)
- real-world case study
- extending direction or future work

### Course Project
[Report](https://www.overleaf.com/read/cvvhfbrykfcr)

[Slides](https://github.com/yul091/QASlow/raw/main/course_project/Presentation.pptx)
