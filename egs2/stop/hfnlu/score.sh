result=$1
if [ -z "${2}" ]; then
    dset="test"
else
    dset=$2
fi

python local/score_sclite.py ${result} --dset ${dset}

expdir=${result%.*}

$PWD/../../../tools/sctk-2.4.10/bin/sclite \
    -r "${expdir}/ref.trn" trn \
    -h "${expdir}/hyp.trn" trn \
    -i rm -o all stdout > "${expdir}/result.txt"

grep -e Avg -e SPKR -m 2 "${expdir}/result.txt"
