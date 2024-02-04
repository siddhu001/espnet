import json
from pathlib import Path


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def parse_text(text: str):
    # text: <0.00> be conceived without god<3.68><9.72> therefore god is the cause of the essence of things<14.84>

    text = text.strip()

    segments = []
    cur_seg = []
    for c in text:
        if c == "<":
            if len(cur_seg) > 0:
                segments.append("".join(cur_seg))
                cur_seg = []
            cur_seg.append(c)
        elif c == ">":
            cur_seg.append(c)
            segments.append("".join(cur_seg))
            cur_seg = []
        else:
            cur_seg.append(c)
    
    assert len(segments) % 3 == 0, segments

    utterances = []
    for idx in range(0, len(segments), 3):
        cur_utt = {
            'start': segments[idx],
            'end': segments[idx + 2],
            'text': segments[idx + 1],
        }
        assert is_float(cur_utt['start'][1:-1]) and is_float(cur_utt['end'][1:-1]), cur_utt
        
        utterances.append(cur_utt)

    return utterances


def parse_owsm(
    text_in,
    text_out,
    suffix=[],
):
    with open(text_in, 'r') as fin, open(
        text_out, 'w'
    ) as fout:
        for line in fin:
            utt_id, text = line.strip().split(maxsplit=1)

            is_valid = False
            for _str in suffix:
                if utt_id.endswith(_str):
                    is_valid = True
                    break
            if not is_valid:
                continue

            if len(text.split("<0.00>")) > 2:
                print(utt_id, text)
                continue

            # prefix: <en><st_ja> or <en><asr>
            prefix, text_part = text.split("<0.00>")

            try:
                utts = parse_text(f"<0.00>{text_part}")
                new_text = f"{prefix}{''.join([u['text'] for u in utts])}"
            except:
                print(utt_id, text_part)
                continue
            
            fout.write(
                f"{utt_id} {new_text}\n"
            )


if __name__ == "__main__":
    root = "dump/raw/MuST-C_v2_en-de"
    suffix = ['en_st_de']
    for name in ['dev', 'train']:
        (Path(root) / name / 'text').rename(Path(root) / name / 'text.old')
        parse_owsm(
            Path(root) / name / 'text.old',
            Path(root) / name / 'text',
            suffix
        )
