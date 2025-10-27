# Random-Logit-Scaling

This is the source code containing the attacks and defenses we used in our experiments. For the attack implementations, we used [BlackboxBench](https://github.com/SCLBD/BlackboxBench) and modified its source code to meet our needs. We implemented our defense, along with the state-of-the-art randomized defenses used for comparison, since BlackboxBench only contains code for attacks and not defenses.

## Running Experiments on CIFAR-10

We have augmented BlackboxBench with the source code of WideResNet-28, VGG-16, and ResNet-18, which we used for training our CIFAR-10 classifiers in the `./models/` directory. We have added code to implement iRND, oRND, AAA, RFD, oRND, and RLS defenses for each model. 

### First Step: Configuring the Defenses

`.\config-jsons\defense_config.json` contains the configuration for the defenses and is used to specify the defense to be used by the victim model. To activate a defense you have to modify the `defense` attribute in the JSON file (lines 2 and 3):
```
"_comment1": "irnd, rfd, ornd, rls, aaa, none",
"defense": "none",
```
For instance, to activate iRND, this should change to:
```
"_comment1": "irnd, rfd, ornd, rls, aaa, none",
"defense": "irnd",
```
The config file then continues with additional attributes for setting the hyperparameter values of each defense:
```
{
    "_comment1": "irnd, rfd, ornd, rls, aaa, none",
    "defense": "none",
    "irnd": {
        "sigma_noise": 0.01
    },
    "rfd": {
        "sigma_noise": 0.7,
        "target_layers": [-1]
    },
    "ornd": {
        "sigma_noise": 1
    },
    "rls":{
        "distribution": "uniform",
        "uniform": {
            "low": 0.5,
            "high": 10
        },
        "gaussian": {
            "low": 0.2,
            "mean": 1,
            "std": 100
        }
    },
    "aaa":{
        "attractor_interval": 4,
        "calibration_loss_weight": 5,
        "dev": 0.5,
        "temperature": 1,
        "optimizer_lr": 0.1,
        "num_iter": 100
    },
    "none": ""
}
```
For example, to use iRND for *v = 0.02* you have to set the `defense` to `irnd` and `sigma_noise` in `irnd` to `0.02`, as shown in the following:
```
{
    "_comment1": "irnd, rfd, ornd, rls, aaa, none",
    "defense": "irnd",
    "irnd": {
        "sigma_noise": 0.02
    },
    "rfd": {
        "sigma_noise": 0.7,
        "target_layers": [-1]
    },
    "ornd": {
        "sigma_noise": 1
    },
    "rls":{
        "distribution": "uniform",
        "uniform": {
            "low": 0.5,
            "high": 10
        },
        "gaussian": {
            "low": 0.2,
            "mean": 1,
            "std": 100
        }
    },
    "aaa":{
        "attractor_interval": 4,
        "calibration_loss_weight": 5,
        "dev": 0.5,
        "temperature": 1,
        "optimizer_lr": 0.1,
        "num_iter": 100
    },
    "none": ""
}
```
#### RFD Configurations

The exact settings that we use for RFD defense is as follows for each model (we target the penultimate layer to adhere to the optimal configuration specified by the authors):

**For VGG:**
```
"rfd": {
        "sigma_noise": 0.35,
        "target_layers": [28]
}
```

**For ResNet-18**:
```
"rfd": {
        "sigma_noise": 2.5,
        "target_layers": [4]
}
```

**For WideResNet-28-10**:
```
"rfd": {
        "sigma_noise": 0.7,
        "target_layers": [3]
}
```

**For ResNet-50**:
```
"rfd": {
        "sigma_noise": 3.0,
        "target_layers": [4]
}
```

### Second Step: Configuring the Attacks

For each black-box attack, a JSON file is provided in the `./config-jsons/` directory to configure the hyperparameter values for that specific attack.

**For CIFAR-10 in L infinity the following config files are provided**
```
./config-jsons/cifar10_bandit_linf_config.json
./config-jsons/cifar10_nes_linf_config.json
./config-jsons/cifar10_sign_linf_config.json
./config-jsons/cifar10_square_linf_config.json
./config-jsons/cifar10_zosignsgd_linf_config.json
```
**For CIFAR-10 in L2**:
```
./config-jsons/cifar10_bandit_2_config.json
./config-jsons/cifar10_nes_2_config.json
./config-jsons/cifar10_square_2_config.json
./config-jsons/cifar10_zosignsgd_2_config.json
```

For instance, the JSON file for SignHunter in L infinity is located at `./config-jsons/cifar10_sign_linf_config.json` and looks like this:
```
{
  "_comment1": "===== DATASET CONFIGURATION =====",
  "dset_name": "cifar10",
  "dset_config": {},
  "_comment2": "===== EVAL CONFIGURATION =====",
  "num_eval_examples": 1000,
  "_comment3": "=====ADVERSARIAL EXAMPLES CONFIGURATION=====",
  "attack_name": "SignAttack",
  "attack_config": {
    "batch_size": 1000,
    "name": "Sign",
    "epsilon": 12.75,
    "p": "inf",
    "fd_eta": 12.75,
    "max_loss_queries": 10000
  },
  "device": "/gpu:1",
  "modeln": "resnet",
  "target": false,
  "target_type": "median",
  "seed": 123
}
```

Common among all of the config files above is the `modeln` attribute which spcifices the victim model and can have the following values for experiments on CIFAR-10:

```json
"modeln": "resnet" ---> ResNet-18
"modeln": "vgg"    ---> VGG-16
"modeln": "wrn"    ---> WideResNet-28-10
```


### Thid Step: Running the Attack

Finally you can run the attack using the following command:

```bash
python attack_cifar10.py [path_to_attack_config]
```

For instance:

```bash
python attack_cifar10.py /path/config-jsons/cifar10_bandit_linf_config.json
```

## Running Experiments on ImageNet

Our implementation for evaluations on ImageNet uses Torchvision's pretrained ResNet-50. Defense implementations for the ResNet-50 are added to the pretrained model using PyTorch hooks (at `./utils/model_loader.py`). 

### First Step: Configuring the Defenses

To configure the defense mode of the victom model, you can modify the JSON file in `./config-jsons/defense_config.json`. The instructions are provided in the CIFAR-10 section.

#### Configuring RFD

**For ResNet-50**:
```
"rfd": {
        "sigma_noise": 3.0,
        "target_layers": [4]
}
```

### Second Step: Configuring the Attacks

The steps are similar to the CIFAR-10 case. Attack config files are as follows:

**For ImageNet in L infinity:**
```
./config-jsons/imagenet_bandit_linf_config.json
./config-jsons/imagenet_nes_linf_config.json
./config-jsons/imagenet_sign_linf_config.json
./config-jsons/imagenet_square_linf_config.json
./config-jsons/imagenet_zosignsgd_linf_config.json
```
**For ImageNet in L2:**
```
./config-jsons/imagenet_bandit_l2_config.json
./config-jsons/imagenet_nes_l2_config.json
./config-jsons/imagenet_square_l2_config.json
./config-jsons/imagenet_zosignsgd_l2_config.json
```

Each config file has a `modeln` attribute that has to be set to:
```
modeln: Resnet50
```


### Third Step: Runing the attack

Instead of `attack_cifar10.py` you have to use `attack_imagenet.py`.

```bash
python attack_imagenet.py [path_to_attack_config]
```

## Adaptive Attack against AAA-sine

We develop an adaptive attack against AAA-sine to show its vulnerability. To run this adaptive attack you have to use square attack and modify the JSON file in `./conifg-jsons/adaptive.json` as follows:

```python
{
    "method": "switch_dir",
    "switch_dir": {
        "k": 10
    },
    "none": {
    },
    "verbose": true
}
```
setting the `method` to `none` disables the adaptive attack. `k` in `switch_dir` specifies the number of unsuccessful attack iterations before the attacker switches direction as explained in Section 4 of the paper.
