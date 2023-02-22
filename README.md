# Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting
This is a implementation of DMSTGCN: [Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting, KDD2021].
## Environment
- python 3.7.4
- torch 1.2.0
- numpy 1.17.2

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

```bash
conda install pandas pytables snappy
```

## Dataset
Step 0： 
```bash
mkdir -p data/{PEMS07,PEMS08,HZME_INFLOW,HZME_OUTFLOW}
```

````bash
python generate_training_data.py --traffic_df_filename data/PEMS07.h5 --output_dir data/PEMS07
python generate_training_data.py --traffic_df_filename data/PEMS08.h5 --output_dir data/PEMS08
python generate_training_data.py --traffic_df_filename data/HZME_INFLOW.h5 --output_dir data/HZME_INFLOW
python generate_training_data.py --traffic_df_filename data/HZME_OUTFLOW.h5 --output_dir data/HZME_OUTFLOW
````

Step 1： Download the processed dataset from [Baidu Yun](https://pan.baidu.com/s/1UpvcgaGp2D-ff80pX65cJA) (Access Code:luck).

If needed, the origin dataset of PEMSD4 and PEMSD8 are available from [ASTGCN](https://github.com/Davidham3/ASTGCN).

Step 2: Put them into data directories.
## Train command
    # Train with PEMSD4
    python train.py --data=PEMSD4
    
    # Train with PEMSD8
    python train.py --data=PEMSD8
    
    # Train with England
    python train.py --data=England

```bash
nohup python train.py --data=PEMS03 --iden PEMS03_01 --cuda 5 > PEMS03-Iter1.log &
nohup python train.py --data=PEMS04 --iden PEMS04_01 --cuda 7 > PEMS04-Iter1.log &
python train.py --data=PEMS07
python train.py --data=PEMS08
python train.py --data=HZME_INFLOW
python train.py --data=HZME_OUTFLOW
```