# CS395T_final_project
This repo host all the source code and documentation regarding the Deep Learning Seminar (CS395T) final project.


HOW TO GET THE SCORES for valication data:

python create_json_references.py -i data/val_captions_with_filenames.txt -o data/val_captions_with_filenames.json


python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_RNN_stacked_LSTM_layers_2.txt -r data/val_captions_with_filenames.json


YOLO9000
python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_9000_RNN_regular_LSTM_layers_1.txt -r data/val_captions_with_filenames.json

BIDIRECTIONAL
python run_evaluations.py -i data/predicted_val_captions_with_filenames_GlobalAvgPool2D_RNN_bidirectional_LSTM_layers_1.txt -r data/val_captions_with_filenames.json



