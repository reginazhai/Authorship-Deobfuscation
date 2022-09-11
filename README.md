# Authorship-Deobfuscation
This repository contains code and data used in the paper "[A Girl Has A Name, And It’s ... Adversarial Authorship Attribution for Deobfuscation][paper-link]" accepted at ACL 2022.

## Requirements
1. Install Python3
2. Download trained models and data for training and testing in the main folder. (https://www.dropbox.com/sh/iyuviaafxx94yda/AAAG4nBL0p6B5f36qsp-1kCQa?dl=0)
3. Install libraries needed for the project through
```
  pip3 install -r requirements.txt
```

## Usage

### Evaluate prediction accuracy for obfuscated texts
To evaluate directly using previous data, run the following in the main folder.
```
  python3 eval.py
```

### Evaluate performance on METEOR score
To evaluate the METEOR score from previous data, run the following in the main folder.
```
  python3 CalcMeteor.py
```


[paper-link]:https://aclanthology.org/2022.acl-long.509/ 
