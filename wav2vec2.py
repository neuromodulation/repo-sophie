import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments, Wav2Vec2Tokenizer, DataCollatorWithPadding
import soundfile as sf
from datasets import load_dataset
import librosa
from sklearn.model_selection import train_test_split
import shutil
import os
import pickle


#before starting the code, u have to $env:PYTHONUTF8="1" in Terminal (apparently only runs with UTF8)

"""
def preprocess_function(batch, processor):
    speech_array, sampling_rate = sf.read(batch["path"])

    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)
    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    labels = processor(text=batch["sentence"],sampling_rate=16000, return_tensors="pt", padding=True).input_ids
    # with processor.as_target_processor():
    #     labels = processor(batch["sentence"], return_tensors="pt").input_ids

    inputs["label"] = labels
    return inputs
    """
def preprocess_function(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
data_collator = DataCollatorWithPadding(tokenizer)

class IterableDATA(IterableDataset):
    def __init__(self, dataset, preprocess_function, processor):
        self.dataset = dataset
        self.preprocess_function = preprocess_function
        self.processor = processor

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield preprocess_function(self.dataset[i], self.processor)

    def __len__(self):
        return len(self.dataset)
        

def main():
    #berlindaten auf 1d tensor runtersampeln

    cache = os.path.expanduser('~\\.cache\\huggingface\\transformers')
    if os.path.exists(cache):
        shutil.rmtree(cache)
        print("Cache has been cleared")
    else:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"trains on {device}")

    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    model = model.to(device)
    
#gitignore: ganzen ordner? wegen transformern & rye container stuff
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True) 
    # dataset = load_dataset("emotion", split="train[:10%]")
    df = dataset.map(preprocess_function)
    inputs = tokenizer(df["speech"], return_tensors="pt", padding=True).input_values.to(device)
    # outputs = model(inputs)
    logits = model(inputs).logits #logits statt hidden states, wenn man mit CTC arbeitet
    #hidden_states = model(inputs).last_hidden_state

    #train_dataset, eval_dataset = train_test_split(df, test_size=0.2, random_state=42)
    #train_ds = dataset.from_pandas(train_dataset)
    #eval_ds = dataset.from_pandas(eval_dataset)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h", trust_remote_code=True)
    #processed_dataset = dataset.map(preprocess_function)
    #train_data = IterableDATA(train_ds, preprocess_function, processor)
    #eval_data = IterableDATA(eval_ds, preprocess_function, processor)

    training_arg = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8, ###
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        save_total_limit=2,
        save_steps=500,
        fp16=False,
        gradient_accumulation_steps=2,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_arg,
        train_dataset=inputs,
        #tokenizer=tokenizer,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./fine_tuned_wav2vec2")
    predicted_ids = torch.argmax(logits, dim=-1)

if __name__ == "__main__":
    # freeze_support()
    #mp.set_start_method("spawn")
    main()