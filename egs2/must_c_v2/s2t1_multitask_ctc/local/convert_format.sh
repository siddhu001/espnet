# python local/convert_format.py



# Copied from utils/fix_data_dir.sh
function check_sorted {
  file=$1
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
}

utt_extra_files="text.prev text.ctc"

# NOTE(yifan): extra text files must be sorted and unique
dev_out="dump/raw/owsm_dev.en-de"
for f in ${utt_extra_files}; do
    check_sorted ${dev_out}/${f}
done
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${dev_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print ${dev_out} || exit 1;

# NOTE(yifan): extra text files must be sorted and unique
train_out="dump/raw/owsm_train.en-de_sp"
for f in ${utt_extra_files}; do
    check_sorted ${train_out}/${f}
done
utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" ${train_out} || exit 1;
utils/validate_data_dir.sh --no-feats --non-print ${train_out} || exit 1;
