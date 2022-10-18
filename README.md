# Energy attack on question answering (Q.A.)

## Quicktart

### Setup Environment
- python3.6+
- install libraries
```
pip install -r requirements.txt
```
- download datasets & word2vec & en module for spacy
```
bash download.sh
```

### Train a model on SQuAD v2.0

- preprocess data
```
python preprocess.py
```

- train a model (e.g., UNet)
```
python train.py --model unet
```

- evaluate a model (e.g., UNet)
```
python train.py --model unet --eval
```

### Attack the pre-trained model
```
python attack.py --model unet --dataset squad
```


