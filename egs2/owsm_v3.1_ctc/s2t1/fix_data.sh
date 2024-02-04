export LC_ALL=C

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


for x in dev_v3 train_v3; do
    check_sorted dump/raw/$x/wav.scp
done
