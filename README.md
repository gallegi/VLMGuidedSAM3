# Project goal
- Use VLM to support robust tracking through occlusion with SAM3

# Installation
- Follow the instruction in the original SAM3 and EasyR1's repositories to install the two libs.
- Additional dependencies are in requirements.txt

# How to run
## Generate SAM3's prediction on SAV training set
```
python scripts/collect_training_data_v2.py --sav_dataset /graphics/scratch3/datasets/sav_train/sav_train/ \
    --sub_dir sav_000 \
    --output_dir /graphics/scratch2/students/nguyenth/SAV/sam3_prediction \
    --sav_train_format \
```
- The script results in a folder with the following structure:
```
sav_000/
├── tracking_result/
│   ├── sav_0000001.json
│   ├── sav_0000002.json
│   └── sav_0000003.json
└── visualization/

sav_001/
├── tracking_result/
│   ├── sav_0000001.json
│   ├── sav_0000002.json
│   └── sav_0000003.json
└── visualization/
```
## Generate VLM-finetuning format
```
python scripts/gen_tracking_verification_data.py \
    --collected_dir /graphics/scratch2/students/nguyenth/SAV/sam3_prediction \
    --sub_dir sav_000 \
    --output_dir /graphics/scratch2/students/nguyenth/SAV/vlm_formatted_data \
    --output_json /graphics/scratch2/students/nguyenth/SAV/vlm_formatted_data/tracking_verification_train.json \
    --max_samples_per_object 2 \
```
- The script results in a folder with the following structure:
```
<outdir>/
├── images/
│   ├── sav_000001_obj1_f288_correct_ctx0.json
│   ├── sav_000001_obj1_f288_correct_ctx1.json
│   ├── sav_000001_obj1_f288_correct_ctx2.json
│   ├── sav_000001_obj1_f288_correct_ctx3.json
│   └── sav_000001_obj1_f288_correct_query.json
├── vis/
└── tracking_verification_train.json
```