import os
from datasets import load_dataset
import json
import numpy as np
import random
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Wav2Vec2CTCTokenizer, Trainer, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from evaluate import load
import soundfile as sf
import re
import torch
from typing import Union, Dict, Optional, List
from dataclasses import dataclass
from read_data2 import BIDSLoader

# if "charmap" error: $env:PYTHONUTF8="1" in Terminal

# without pretrained model
# runs through and continues training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"trains on {device}")

def main():
    model_dir = "repo-sophie-1/wav2vec2-without-pretraining-demo"

    timit = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", trust_remote_code=True)
    timit = timit.remove_columns(["speaker_id", "id", "chapter_id"])
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
    vocab_list = list(set(vocabs["validation"]["vocab"][0]) | set(vocabs["validation"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                                     sampling_rate=16000, 
                                                     padding_value=0.0, 
                                                     do_normalize=True, 
                                                     return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    try:
        if os.path.isdir(model_dir):
            print(f"loading model from {model_dir}...")
            model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
        else:
            raise ValueError(f"no model found {model_dir}")
    
    except Exception as e:
        print(f"{e} starting training new")
        config = Wav2Vec2Config(vocab_size=len(vocab_dict), 
                                ctc_loss_reduction="mean",
                                pad_token_id=processor.tokenizer.pad_token_id)
        model = Wav2Vec2ForCTC(config).to(device)
    
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch

    timit = timit.map(speech_file_to_array_fn, num_proc=4)

    rand_int = random.randint(0, len(timit["validation"]))

    print("Input array shape:", np.asarray(timit["validation"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", timit["validation"][rand_int]["sampling_rate"])

    def prepare_dataset(batch):
        speech_array = batch["audio"]["array"]
        sampling_rate = batch["audio"]["sampling_rate"]
        inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        batch["input_values"] = inputs.input_values[0]
        batch["attention_mask"] = inputs.attention_mask[0]
        batch["target_text"] = batch["text"]

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    # timit_prepared = timit.map(prepare_dataset)
    train_test_split = timit["validation"].train_test_split(test_size=0.2)
    train_data = train_test_split["train"]
    test_data = train_test_split["test"]

    train_data = train_data.map(prepare_dataset, remove_columns=timit["validation"].column_names)
    test_data = test_data.map(prepare_dataset, remove_columns=timit["validation"].column_names)


    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load("wer")
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    training_args = TrainingArguments(
        output_dir=model_dir,
        group_by_length=False,
        per_device_train_batch_size=32,
        eval_strategy="steps",
        num_train_epochs=20,
        fp16=False,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.evaluate()

    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

if __name__ == "__main__":
    main()
