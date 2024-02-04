data_dir="dump/raw/MuST-C_v2_en-de/tst-COMMON.en-de"

# cut -d " " -f 1 ${data_dir}/text > ${data_dir}/_uttids
# cut -d " " -f 2- ${data_dir}/text > ${data_dir}/_text

# detokenizer.perl -l en -q < ${data_dir}/_text > ${data_dir}/_text.detok

# mv ${data_dir}/text ${data_dir}/text.tok
# paste -d " " ${data_dir}/_uttids ${data_dir}/_text.detok > ${data_dir}/text

# cut -d " " -f 2- ${data_dir}/text.tc.de > ${data_dir}/_text.tc.de
# detokenizer.perl -l de -q < ${data_dir}/_text.tc.de > ${data_dir}/_text.tc.de.detok

mv ${data_dir}/text.tc.de ${data_dir}/text.tc.de.tok
paste -d " " ${data_dir}/_uttids ${data_dir}/_text.tc.de.detok > ${data_dir}/text.tc.de
