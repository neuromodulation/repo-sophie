import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
import soundfile as sf
from datasets import load_dataset
import librosa
from sklearn.model_selection import train_test_split
import shutil
import os

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

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h", trust_remote_code=True)

def preprocess_function(batch):
    speech_array, sampling_rate = sf.read(batch["path"])

    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt").input_ids

    inputs["labels"] = labels
    return inputs

def main():

#gitignore: ganzen ordner? wegen transformern & rye container stuff
#is now training on english, "ru" for russian, "de" for german
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train[:2%]", trust_remote_code=True) 
    df = dataset.to_pandas()
    train_dataset, eval_dataset = train_test_split(df, test_size=0.2, random_state=42)
    train_ds = dataset.from_pandas(train_dataset)
    eval_ds = dataset.from_pandas(eval_dataset)

#processed_dataset = dataset.map(preprocess_function)
    train_data = train_ds.map(preprocess_function)
    eval_data = eval_ds.map(preprocess_function)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    model = model.to(device)

    training_arg = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8, ###
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        save_total_limit=2,
        save_steps=500,
        fp16=True,
        gradient_accumulation_steps=2,
        dataloader_num_workers=4
    )

    trainer = Trainer(
        model=model,
        args=training_arg,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=processor.feature_extractor
    )

    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./fine_tuned_wav2vec2")

if __name__ == "__main__":
    # freeze_support()
    main()