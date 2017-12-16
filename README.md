CS395T_final_project
This repo host all the source code and documentation regarding the Deep Learning Seminar (CS395T) final project.


HOW TO GET THE SCORES for valication data:

python create_json_references.py -i data/val_captions_with_filenames.txt -o data/val_captions_with_filenames.json


python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_RNN_stacked_LSTM_layers_2.txt -r data/val_captions_with_filenames.json


YOLO9000
python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_9000_RNN_regular_LSTM_layers_1.txt -r data/val_captions_with_filenames.json



GlobalAveragePooling2D, regular 
python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_RNN_regular_LSTM.txt -r data/val_captions_with_filenames.json


GlobalAveragePool2D_last, regular
python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_last_RNN_regular_LSTM_layers_1.txt -r data/val_captions_with_filenames.json

AveragePooling2D
python run_evaluations.py -i data/predicted_val_captions_with_filenames_AveragePooling2D_RNN_regular_LSTM_layers_1.txt -r data/val_captions_with_filenames.json

HOW TO USE YAD2K to perform bounding box detection:
in the main directory run this:

CUDA_VISIBLE_DEVICES=1 ./test_yolo.py model_data/yolo.h5 -t="images/in_selected" -o="images/out_selected_imgs"


