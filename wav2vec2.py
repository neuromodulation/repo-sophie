import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments, Wav2Vec2Tokenizer
import soundfile as sf
from datasets import load_dataset
import librosa
from sklearn.model_selection import train_test_split
import shutil
import os
import pickle


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
    #
    #before starting the code, u have to $env:PYTHONUTF8="1" in Terminal (apparently only runs with UTF8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"trains on {device}")

    try:
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        model = model.to(device)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"failed to load model: {e}###########################################################")

#gitignore: ganzen ordner? wegen transformern & rye container stuff
#is now training on english, "ru" for russian, "de" for german
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True) 
    #try:
    #    dataset = load_dataset("emotion", split="train[:10%]")
    df = dataset.map(preprocess_function)
    inputs = tokenizer(df["speech"][0], return_tensors="pt").input_values 
    hidden_states = model(inputs).last_hidden_state
    #except (pickle.UnpicklingError, EOFError) as e:
    #    print(f"failed to load Dataset: {e}########################################################")

    #train_dataset, eval_dataset = train_test_split(df, test_size=0.2, random_state=42)
    #train_ds = dataset.from_pandas(train_dataset)
    #eval_ds = dataset.from_pandas(eval_dataset)

    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h", trust_remote_code=True)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"failed to load processor: {e}###########################################################")

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
        tokenizer=processor.feature_extractor
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./fine_tuned_wav2vec2")

if __name__ == "__main__":
    # freeze_support()
    #mp.set_start_method("spawn")
    main()