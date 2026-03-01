# run HSJA signal, example for audioseal
python3 black-box-HSJA_signal.py --gpu 0 --testset_size 200  --query_budget 10000  --tau 0.15 --norm linf --model audioseal --blackbox_folder audioseal_samp 

# run HSJA spectrogram, exmaple for audioseal 
python3 black-box-HSJA_spectrogram.py --gpu 0 --testset_size 200  --query_budget 10000  --tau 0.15 --norm linf --model audioseal --blackbox_folder audioseal_samp --attack_type both

# run square attack, example for audioseal
python3 black-box_square.py --gpu 0 --testset_size 200  --query_budget 10000  --tau 0.15 --model audioseal --blackbox_folder audioseal_samp --attack_type both --eps 0.05
