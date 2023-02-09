from num2words import num2words
file1=open("dump/raw/test_asr_whisper/transcript")
file1_write=open("dump/raw/test_asr_whisper/transcript_new","w")
for line in file1:
    arr1=[line.split()[0]]
    for s in line.split()[1:]:
        if s.isdigit():
            arr1.append(num2words(s).upper())
        else:
            arr1.append(s.upper())
    file1_write.write(" ".join(arr1)+"\n")