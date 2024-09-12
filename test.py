from datasets import load_dataset_builder, load_dataset
import librosa
from transformers import Wav2Vec2Processor
import soundfile as sf
from sklearn.model_selection import train_test_split

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
load_dataset_builder("mozilla-foundation/common_voice_11_0")

def preprocess_function(batch):
    speech_array, sampling_rate = sf.read(batch["path"])

    if sampling_rate != 16000:
        speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=16000)

    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

    with processor.as_target_processor():
        labels = processor(batch["sentence"], return_tensors="pt").input_ids

    inputs["labels"] = labels
    return inputs

load_dataset_builder()
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", trust_remote_code=True) 
df = dataset.to_pandas()
train_dataset, eval_dataset = train_test_split(df, test_size=0.2, random_state=42)
train_ds = dataset.from_pandas(train_dataset)
eval_ds = dataset.from_pandas(eval_dataset)

#processed_dataset = dataset.map(preprocess_function)
train_data = train_ds.map(preprocess_function)
eval_data = eval_ds.map(preprocess_function)