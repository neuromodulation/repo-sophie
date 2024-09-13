from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2Tokenizer
from datasets import load_dataset
import soundfile as sf
import torch

tokenizer = Wav2Vec2Tokenizer("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model("facebook/wav2vec2-base-960h")

def preprocessing(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)
input_values = tokenizer(ds["speech"][0], return_tensors="pt").input_values
#hidden_states = model(input_values).last_hidden_state
logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])