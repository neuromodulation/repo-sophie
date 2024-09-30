import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Tokenizer, Wav2Vec2ForCTC, TrainingArguments, Wav2Vec2FeatureExtractor, Wav2Vec2Config, Trainer
from evaluate import load
from bids_load import BIDSBrainVisionDataset
from dataclasses import dataclass
from typing import Union, Dict, Optional, List
import json
import numpy

# if "charmap" error: $env:PYTHONUTF8="1" in Terminal
#w/ pretrained model, 4 now unsupervised
#maybe contrastive loss as lossfunc
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (2, 1) + inhomogeneous part.
#in generic.py, line 299, this operation fails, cuz obj is a tuple of 2 tensors without the same shape. maybe i have to modify the whole structure :(
# unlikely that it works with dtype=object (cuz numerical ops)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")

def multichannel_W2V2(model, num_channels=6):
    old_conv = model.wav2vec2.feature_extractor.conv_layers[0].conv

    new_conv = torch.nn.Conv1d(
        in_channels=num_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    model.wav2vec2.feature_extractor.conv_layers[0].conv = new_conv
    return model

def main():
    model_dir = "wav2vec2-demo"
    channel_names = ['ECOG_RIGHT_0', 'ECOG_RIGHT_1', 'ECOG_RIGHT_2', 'ECOG_RIGHT_3', 'ECOG_RIGHT_4', 'ECOG_RIGHT_5']
    target_name = None
    dataset = BIDSBrainVisionDataset(
        directory="data",
        channel_names=channel_names,
        target_name=target_name,
        window_size=2.0,
        overlap=0.0
    )
    vocab_dict = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    with open("dummy_vocab.json", 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2Tokenizer("dummy_vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")    
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    config = Wav2Vec2Config(
        vocab_size=len(vocab_dict),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id
    )

    model = Wav2Vec2ForCTC(config).to(device)
    model = multichannel_W2V2(model, num_channels=len(channel_names)).to(device)

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

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
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

if __name__ == "__main__":
    main()
