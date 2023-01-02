# Energy attack on Dialogue Generation

## Quickstart

### Setup Environment
- python 3.10.8
- pytorch 1.13.0+
- Install dependencies
```
pip install -r requirements.txt
```

### Train and evaluate a model on a specific task(s)

- BART on BlendedSkillTalk:
```
python train_seq2seq.py --model_name_or_path facebook/bart-base --dataset blended_skill_talk --output_dir results/bart
```
- DialoGPT on EmpatheticDialogues:
```
python train_clm.py --model_name_or_path microsoft/DialoGPT-small --dataset empathetic_dialogues --output_dir results/dialogpt
```
- Or you can directly download our pre-trained [model](https://drive.google.com/drive/folders/1rWexrwHCgCFYiNVk2yFKSI8iV8baWfFt?usp=sharing) 

### Attack a pre-trained model
- Structure attack on BART on BlendedSkillTalk dataset
```
python attack.py --attack_strategy structure --model_name_or_path results/bart --dataset blended_skill_talk
```
- Word attack on DialoGPT on EmpatheticDialogues dataset 
```
python attack.py --attack_strategy word --model_name_or_path microsoft/DialoGPT-small --dataset empathetic_dialogues
```
- PWWS (baseline) on T5 on ConvAI2 dataset
```
python attack.py --attack_strategy pwws --model_name_or_path results/t5 --dataset conv_ai_2
```
- VIPER (baseline) on T5 on PersonaChat dataset
```
python attack.py --attack_strategy viper --model_name_or_path results/t5 --dataset AlekseyKorshuk/persona-chat
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

<!-- 
### Course Project
[Report](https://www.overleaf.com/read/cvvhfbrykfcr)

[Slides](https://github.com/yul091/QASlow/raw/main/course_project/Presentation.pptx) -->
