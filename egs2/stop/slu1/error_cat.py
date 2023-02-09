file_hyp_bart=open("/projects/bbjs/arora1/espnet/egs2/stop/slu1/exp/slu_train_asr2_wavlm_bart3_whisper_loss3_raw_en_bpe500_sp/decode_asr9_lm_lm_train_rnn_lm_en_bpe500_valid.loss.ave_slu_model_valid.acc.ave_10best/test_asr_whisper/postdec_out")
file_hyp=open("/projects/bbjs/arora1/espnet/egs2/stop/slu1/exp/slu_train_asr2_wavlm_bart3_whisper_loss3_raw_en_bpe500_sp/decode_asr9_lm_lm_train_rnn_lm_en_bpe500_valid.loss.ave_slu_model_valid.acc.ave_10best/test_asr_whisper/text")
file_ref=open("dump/raw/test/text")
line_bart=[line for line in file_hyp_bart]
line_hyp=[line for line in file_hyp]
line_ref=[line for line in file_ref]
file_write=open("ref_bart_error.txt","w")
file_write1=open("ref_hyp_error.txt","w")
for line_count in range(len(line_ref)):
    if line_ref[line_count]==line_bart[line_count]:
        if line_ref[line_count]!=line_hyp[line_count]:
            file_write.write(line_ref[line_count].strip()+" "+line_hyp[line_count])
for line_count in range(len(line_ref)):
    if line_ref[line_count]==line_hyp[line_count]:
        if line_ref[line_count]!=line_bart[line_count]:
            file_write1.write(line_ref[line_count].strip()+" "+line_bart[line_count])