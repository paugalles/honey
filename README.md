# Honey

This is an exercice of honey microscope image classification.

## Training

usage: train.py [-h] [--split_random_seed SPLIT_RANDOM_SEED]
                [--batch_sz BATCH_SZ] [--n_epochs N_EPOCHS]
                [--num_workers NUM_WORKERS]
                [--last_layer_num_neurons LAST_LAYER_NUM_NEURONS]
                [--learning_rate LEARNING_RATE] [--experiment EXPERIMENT]
                [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --split_random_seed SPLIT_RANDOM_SEED
                        split_random_seed
  --batch_sz BATCH_SZ   batch_sz
  --n_epochs N_EPOCHS   n_epochs
  --num_workers NUM_WORKERS
                        num_workers
  --last_layer_num_neurons LAST_LAYER_NUM_NEURONS
                        in case you want to set the number of layers for the
                        first flat layer, specified as a comma separated
                        string such as "64,128,256"
  --learning_rate LEARNING_RATE
                        learning_rate
  --experiment EXPERIMENT
                        experiment_name for mlflow
  --model MODEL         model name

## Inference & Testing

usage: test.py [-h] [--weightsfn WEIGHTSFN] [--outpath OUTPATH]
               [--model MODEL] [--split_random_seed SPLIT_RANDOM_SEED]
               [--batch_sz BATCH_SZ] [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --weightsfn WEIGHTSFN
                        weights filename
  --outpath OUTPATH     outpath for the inference results
  --model MODEL         model name
  --split_random_seed   SPLIT_RANDOM_SEED
                        split_random_seed
  --batch_sz BATCH_SZ   batch size
  --num_workers NUM_WORKERS
                        num_workers

## Installation & environemnt

Create a python environment, then install the requirements with:

```bash
pip3 install -r requirements.txt
```

Alternatively you can use docker and docker as follows:

1. Build the docker image

```bash
make build
```

2. Raise a running container

```bash
make container
```

3. As soon as the docker is running in background you can launch the following services whithin it:

    * Launch a jupyter lab server

        ```bash
        make nb
        ```

        Then access it from your browser by using this address `localhost:8088/?token=hny`

    * Stops the jupyter server

        ```bash
        make nbstop
        ```

    * Launch an mlflow server

        ```bash
        make mlf
        ```
        Then access it from your browser by using this address `localhost:5055`

    * To raise an interactive shell from our running container

        ```bash
        make execsh
        ```
        
    * To run tests

        ```bash
        make test
        ```

    * Stop the docker container and everything that is running within it

        ```bash
        make stop
        ```