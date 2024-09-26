import re
import jiwer
import os

inaudible_tags = ['[music] [inaudible]', '(inaudible) ', '[inaudible)', '(inaudible]',
                  '[Inaudible].', '[music]','[INAUDIBLE]',' [Inaudible]', '(Inaudible).',
                  '[Inaudible] ', '[silence]','[Silence]', '[inaudible] ', 'in aduible',
                  '(inaudible)','(Inaudible)','[Inaudible]', 'Inaudible','[inaudible]',
                  '[inaudable]','[Inaudible]','Inaudable ','Blank ', 'inaudible', 'Inaudible ', 
                  '(audio is empty)', 'noise', '(noise)', '[noise]', 'Blank'
                 ]
inaudible_tags_regex = [x.replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)') for x in inaudible_tags]
inaudible_tags_joined = "|".join(inaudible_tags_regex)
general_filler_words = ["ah", "blah", "eh", "hmm", "huh", "hum", "mmhmm", "mm", "oh", "ohh", "uh", "uhhuh", "umhum", "uhhum", "um"]


def clean_text(text):
    if type(text) != str:
        print(text)
        return " "

    # remove multiple spaces
    text = clean_filler_words(text)
    text = re.sub(r"\s\s+", " ", text)
    # strip trailing spaces
    text = text.strip()
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = text.lower()
    text = text.replace(" comma,", ",") \
        .replace(" koma,", " ") \
        .replace(" coma,", ",") \
        .replace(" comma", " ") \
        .replace(" full stop.", ".") \
        .replace(" full stop", ".") \
        .replace(",.", ".") \
        .replace(",,", ",") \
        .strip()
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    return text



def clean_filler_words(text):
    text = text.replace("inaudible. ", "").replace("inaudible", "")\
        .replace(" ehm, ", " ").replace(" uh, "," ").replace(" er, "," ").replace("...", " ")
    
    tokens = re.findall(r'\b\w+\b', text)
    cleaned_tokens = [token for token in tokens if token not in general_filler_words]
    return ' '.join(cleaned_tokens)

def post_process_preds(data):
    assert "hypothesis" in data.columns
    assert "reference" in data.columns

    pred_clean = [clean_text(text) for text in data["hypothesis"]]
    ref_clean = [clean_text(text) for text in data["reference"]]

    pred_clean = [text if text != "" else "abcxyz" for text in pred_clean]
    ref_clean = [text if text != "" else "abcxyz" for text in ref_clean]

    data["pred_clean"] = pred_clean
    data["ref_clean"] = ref_clean

    data["wer"] = data.apply(lambda row: jiwer.wer(row["ref_clean"], row["pred_clean"]), axis=1)

    all_wer = jiwer.wer(list(data["ref_clean"]), list(data["pred_clean"]))
    print(f"WER: {all_wer * 100:.2f} %")

    return all_wer