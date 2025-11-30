python eval.py --model_path models/roberta_scratch_full/final --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/roberta_base_full_submission.csv
python eval.py --model_path models/codebert_full/final --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/codebert_base_submission.csv
python eval.py --model_path models/unixcoder_full/final --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/unixcoder_base_submission.csv
python eval.py --model_path models/unixcoder_focal/checkpoint-10000 --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/unixcoder_focal_submission.csv
python eval.py --model_path models/unixcoder_optimized/final --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/unixcoder_optimized_submission.csv
python eval.py --model_path models/modernbert_base_full/final --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/modernbert_base_submission.csv
python eval.py --model_path models/modernbert_augmentation_full/final --max_length 510 --do_eval --batch_size 128 --do_predict --submission_file predictions/modernbert_augmentation_submission.csv
python eval.py --model_path models/modernbert_longer_full/checkpoint-15625 --max_length 1024 --do_eval --batch_size 128 --do_predict --submission_file predictions/modernbert_longer_submission.csv
python eval.py --model_path models/modernbert_extra_longer_full/checkpoint-1000 --max_length 2048 --do_eval --batch_size 64 --do_predict --submission_file predictions/modernbert_extra_longer_submission.csv


