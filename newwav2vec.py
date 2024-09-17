from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2Tokenizer, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import soundfile as sf
import torch
from datasets import load_dataset#, load_metric
import json
import re

from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML

# def show_random_elements(dataset, num_examples=10):
#     assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
#     picks = []
#     for _ in range(num_examples):
#         pick = random.randint(0, len(dataset)-1)
#         while pick in picks:
#             pick = random.randint(0, len(dataset)-1)
#         picks.append(pick)
    
#     df = pd.DataFrame(dataset[picks])
#     display(HTML(df.to_html()))



def main():

    timit = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", trust_remote_code=True)
    timit = timit.remove_columns(["file", "speaker_id", "id", "chapter_id"])
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    def remove_special_characters(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
        return batch
    timit = timit.map(remove_special_characters)

    def extract_all_chars(batch):
        all_text = " ".join(batch["text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}



    vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["validation"])
    vocab_list = list(set(vocabs["validation"]["vocab"][0]) | set(vocabs["validation"]["vocab"][0])) #eigentlich mit train & test
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
#print(vocab_dict)
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
#30 tokens in dict (or 28 in our case), so liniear layer with output dimension 28 on top of pretrained

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

#feature extractor:

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch
    timit = timit.map(speech_file_to_array_fn, num_proc=4)

if __name__ == "__main__":
    main()
