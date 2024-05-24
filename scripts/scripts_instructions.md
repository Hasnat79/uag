

## experiment scripts
```
├── videochat2_x_uag_oops_v1.py
├── video_chatgpt_x_ssbd.py
├── video_chatgpt_x_uag_oops_v1.py
├── video_llama2_x_ssbd.py
└── video_llama2_x_uag_oops_v1.py
```
## Data loaders
```
├── ssbd_loader.py
└── uag_oops_v1_loader.py
```
## Model loaders
```
├── videochat2_loader.py
├── video_chatgpt_loader.py
├── video_llama2_loader.py
└── video_llama2_uag_oops_Out.10429403
```

## Results
- the results are saved in [./results](./results) folder
## Evaluation
- the evaluation scores are generated and saved in [./eval](./eval) folder
## dataset prep
- `oops_data_builder.py` -- creates uag_oops_dataset_v1.json
- datset is saved inside [./data/oops_dataset/uag_oops_datset](./data/oops_dataset/uag_oops_datset/) folder
