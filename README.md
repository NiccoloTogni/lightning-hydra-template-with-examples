## Deep learning project template
A convenient starting template for most deep learning projects. Built with <b>PyTorch Lightning</b> and <b>Weights&Biases</b>.<br>


## Features
- Predefined folder structure
- Storing project configuration in a convenient way ([project_config.yaml](project/project_config.yaml))
- Storing many run configurations in a convenient way ([run_configs.yaml](project/run_configs.yaml))
- All advanteges of Pytorch Lightning
- Automates initialization of your model and datamodule
- Automatically stores code, configurations and model checkpoints in Weights&Biases runs
- Hyperparameter search with Weights&Biases sweeps ([execute_sweep.py](project/utils/execute_sweep.py))
- Built in requirements ([requirements.txt](requirements.txt))
- Built in conda environment initialization ([conda_env.yaml](conda_env.yaml))
- Example with MNIST digits classfication
<br>


## Project structure
The directory structure of new project looks like this: 
```
├── project
│   ├── data                    <- Data from third party sources
│   │
│   ├── logs                    <- Logs generated by Weights&Biases and PyTorch Lightning
│   │
│   ├── notebooks               <- Jupyter notebooks
│   │
│   ├── utils                   <- Different utilities
│   │   ├── callbacks.py            <- Useful training callbacks
│   │   ├── execute_sweep.py        <- Special file for executing Weights&Biases sweeps
│   │   ├── init_utils.py           <- Useful initializers
│   │   └── predict_example.py      <- Example of inference with trained model 
│   │
│   ├── data_modules            <- All your data modules should be located here!
│   │   ├── example_datamodule      <- Each datamodule should be located in separate folder!
│   │   │   ├── datamodule.py           <- Contains 'DataModule' class
│   │   │   ├── datasets.py             <- Contains pytorch 'Dataset' classes
│   │   │   └── transforms.py           <- Contains data transformations
│   │   ├── ...
│   │   └── ...
│   │
│   ├── models                  <- All your models should be located here!
│   │   ├── example_model           <- Each model should be located in separate folder!
│   │   │   ├── lightning_module.py     <- Contains 'LitModel' class with train/val/test step methods
│   │   │   └── models.py               <- Model architectures used by lightning_module.py
│   │   ├── ...
│   │   └── ...
│   │
│   ├── project_config.yaml     <- Project configuration
│   ├── run_configs.yaml        <- Configurations of different runs/experiments
│   └── train.py                <- Train model with chosen run configuration
│
├── .gitignore
├── LICENSE
├── README.md
├── SETUP.md
├── conda_env.yaml
└── requirements.txt
```
<br>


## Project config parameters ([project_config.yaml](project/project_config.yaml))
```yaml
num_of_gpus: -1             <- '-1' to use all gpus available, '0' to train on cpu

loggers:
    wandb:
        project: "project_name"     <- wandb project name
        team: "kino"                <- wandb entity name
        log_model: True             <- set True if you want to upload ckpts to wandb automatically
        offline: False              <- set True if you want to store all data locally

callbacks:
    checkpoint:
        monitor: "val_acc"      <- name of the logged metric that determines when model is improving
        save_top_k: 1           <- save k best models (determined by above metric)
        save_last: True         <- additionaly always save model from last epoch
        mode: "max"             <- can be "max" or "min"
    early_stop:
        monitor: "val_acc"      <- name of the logged metric that determines when model is improving
        patience: 100           <- how many epochs of not improving until training stops
        mode: "max"             <- can be "max" or "min"

printing:
    progress_bar_refresh_rate: 5    <- refresh rate of training bar in terminal
    weights_summary: "top"          <- print summary of model (alternatively "full")
    profiler: False                 <- set True if you want to see execution time profiling
```
<br>


## Run config parameters ([run_configs.yaml](project/run_configs.yaml))
You can store many run configurations in this file.<br>
Example run configuration:
```yaml
MNIST_CLASSIFIER_V1:
    trainer:                                            <- lightning 'Trainer' parameters (all except 'max_epochs' are optional)
        max_epochs: 5                                       
        gradient_clip_val: 0.5                              
        accumulate_grad_batches: 3                          
        limit_train_batches: 1.0                            
    model:                                              <- all of the parameters here will be passed to 'LitModel' in 'hparams' dictionary
        model_folder: "simple_mnist_classifier"             <- name of folder from which 'lightning_module.py' (with 'LitMdodel' class) will be loaded
        lr: 0.001                                           
        weight_decay: 0.000001                              
        input_size: 784                                     
        output_size: 10                                     
        lin1_size: 256                                      
        lin2_size: 256                                      
        lin3_size: 128                                      
    dataset:                                            <- all of the parameters here will be passed to 'DataModule' in 'hparams' dictionary
        datamodule_folder: "mnist_digits_datamodule"        <- name of folder from which 'datamodule.py' (with 'DataModule' class) will be loaded
        batch_size: 256                                     
        train_val_split_ratio: 0.9                          
        num_workers: 1                                      
        pin_memory: False
    wandb:                                              <- this section is optional and can be removed
        group: ""
        tags: ["v2", "uwu"]
    resume_training:                                    <- this section is optional and can be removed if you don't want to resume training
        checkpoint_path: "path_to_checkpoint/last.ckpt"     <- path to checkpoint
        wandb_run_id: None                                  <- you can set id of Weights&Biases run that you want to resume but it's optional                        
```
<br>


## Workflow
1. Add your model to `project/models` folder<br>
    (you need to create folder with `lightning_module.py` file containing `LitModel` class)
2. Add your datamodule to `project/data_modules` folder<br>
    (you need to create folder with `datamodule.py` file containig `DataModule` class)
3. Add your run config to [run_configs.yaml](project/run_configs.yaml) (specify there folders containing your model and datamodule)
3. Configure [project_config.yaml](project/project_config.yaml)
4. Run training:<br>
    Either pass run config name as an argument:
    ```
    python train.py -c MNIST_CLASSIFIER_V1
    python train.py --conf_name MNIST_CLASSIFIER_V1
    ```
   Or modify default run config name in [train.py](project/train.py):
    ```python
    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("-r", "--run_conf_name", type=str, default="MNIST_CLASSIFIER_V1")
        parser.add_argument("-u", "--use_wandb", type=bool, default=True)
        args = parser.parse_args()

        main(run_config_name=args.run_conf_name, use_wandb=args.use_wandb)
    ```
<br><br>


### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

</div>

## Description   
What it does   

## How to run
First, install dependencies
```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment
conda update conda
conda env create -f conda_env.yaml -n your_env_name
conda activate your_env_name

# install requirements
pip install -r requirements.txt
```

Next, you can train model without data logging
```bash
# train model with chosen run configuration
cd project
python train.py --use_wandb=False --run_conf_name MNIST_CLASSIFIER_V1
```

Or you can train model with Weights&Biases data logging 
```yaml
# set project and enity names in project/project_config.yaml
loggers:
    wandb:
        project: "your_project_name"
        entity: "your_wandb_user_or_team_name"
```
```bash
# log to your wandb account
wandb login
```
```bash
# train model with chosen run configuration
cd project
python train.py --use_wandb=True --run_conf_name MNIST_CLASSIFIER_V1
```
<br>

All run configurations are located in [run_configs.yaml](project/run_configs.yaml).

For PyCharm setup read [SETUP.md](SETUP.md).
