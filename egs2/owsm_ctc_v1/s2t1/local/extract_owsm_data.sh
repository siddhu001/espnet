# python local/extract_owsm_data.py

utt_extra_files="text.prev text.ctc"
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" dump/raw/MuST-C_v2/dev
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" dump/raw/MuST-C_v2/train
