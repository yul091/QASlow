# Energy attack on Dialogue Generation

## Quickstart

### Setup Environment
- python3.8+
- install ParlAI:
ParlAI currently requires Python3.8+ and Pytorch 1.6 or higher. Dependencies of the core modules are listed in `requirements.txt`. To install all dependencies, run the following:
```
pip install parlai
```

### Train and evaluate a model on a specific task(s)
All needed data will be downloaded to `data/` folder. If you need to clear out the space used by these files, you can safely delete these directories, and any files required will be downloaded again.

- Seq2seq model on PersonaChat dataset:
```
parlai train_model -t personachat -m seq2seq -mf results/personachat/seq2seq -bs 32 --num-epochs 20
```

- Transformer model on Blended_skill_talk dataset:
```
parlai train_model -t blended_skill_talk -m transformer/generator -mf results/blended_skill_talk/transformer/generator --n-layers 3 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 20 -bs 32 --dropout 0.1 --embedding-type fasttext_cc
```

- Evaluate an IR baseline model on the validation set of the Personachat task:
```
parlai eval_model -m ir_baseline -t personachat -dt valid
```

### Attack a pre-trained model
```
python attack.py --m seq2seq --t personachat
```

### TODO
- :white_check_mark: prepare models
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
