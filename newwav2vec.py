from transformers import Wav2Vec2ForCTC, Trainer, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments
import soundfile as sf
import torch
from datasets import load_dataset
from evaluate import load
import json
import re
import numpy as np
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

#code is running, loaded model isnt finetuned
def main():
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
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch
    timit = timit.map(speech_file_to_array_fn, num_proc=4)

    rand_int = random.randint(0, len(timit["validation"]))

    # print("Target text:", timit["validation"][rand_int]["target_text"])
    print("Input array shape:", np.asarray(timit["validation"][rand_int]["audio"]["array"]).shape)
    print("Sampling rate:", timit["validation"][rand_int]["sampling_rate"])

    def prepare_dataset(batch):
        speech_array = batch["audio"]["array"]
        sampling_rate = batch["audio"]["sampling_rate"]
    
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
    
        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
        return batch

    # timit_prepared = timit.map(prepare_dataset, remove_columns=timit.column_names["validation"], batch_size=8, num_proc=4, batched=True)
    timit_prepared = timit.map(prepare_dataset)
    print(timit_prepared)

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """
            
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
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

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", gradient_checkpointing=True, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id)
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        # output_dir="/content/gdrive/MyDrive/wav2vec2-base-timit-demo",
        output_dir="./wav2vec2-base-timit-demo",
        group_by_length=True,
        per_device_train_batch_size=32,
        eval_strategy="steps",
        num_train_epochs=30,
        fp16=False,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        #gradient_checkpointing=True
        )
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit_prepared["validation"], #i kbnow ur supposed to split the data, but we just testing the hardware for now
        eval_dataset=timit_prepared["validation"],
        tokenizer=processor.feature_extractor,
    )
    
if __name__ == "__main__":
    main()
