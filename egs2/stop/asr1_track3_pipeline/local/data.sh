#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${STOP}" ]; then
    log "Fill the value of 'STOP' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${STOP}/manifests/train.tsv" ]; then
	echo "stage 1: Download data to ${STOP}"
    else
        log "stage 1: ${STOP} is already existing. Skip data downloading"
    fi
    if [ ! -e "${STOPLR}/held_in_train.tsv" ]; then
	echo "Download data to ${STOPLR}"
    else
        log "${STOPLR} is already existing. Skip data downloading"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"
    mkdir -p data/{train-full,valid-full,test-full}
    
    python3 local/prepare_stop_data.py ${STOP}

    for x in test-full valid-full train-full; do
        echo "sort ${x}"
        for f in text transcript wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done

    mkdir -p data/{train-heldin,valid-heldin,test-heldin}
    mkdir -p data/{train-reminder-25spis,valid-reminder-25spis,test-reminder}  # reminder
    mkdir -p data/{train-weather-25spis,valid-weather-25spis,test-weather}  # weather

    python3 local/split_low_resource.py ${STOPLR}

    mkdir -p data/{train,valid,test}

    for f in text transcript utt2spk wav.scp; do
        cat data/train-heldin/${f} data/train-reminder-25spis/${f} data/train-weather-25spis/${f} > "data/train/${f}"
        cat data/valid-heldin/${f} data/valid-reminder-25spis/${f} data/valid-weather-25spis/${f} > "data/valid/${f}"
        cat data/test-heldin/${f} data/test-reminder/${f} data/test-weather/${f} > "data/test/${f}"
    done

    for x in train-heldin valid-heldin test-heldin train-reminder-25spis valid-reminder-25spis test-reminder train-weather-25spis valid-weather-25spis test-weather train valid "test"; do
        echo "sort ${x}"
        for f in text transcript wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
            wc -l data/${x}/${f}
            # 25spis data has duplicate line
            uniq data/${x}/${f} > data/${x}/_${f}
            mv data/${x}/_${f} data/${x}/${f}
            wc -l data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done

    cat data/train/text data/train/transcript > data/train/bpe_text
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
