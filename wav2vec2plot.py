import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from wav2vec2 import *

logging_dir = "./logs"

def load_logs(logging_dir):
    logs = []
    for x in os.listdir(logging_dir):
        if x.endswith(".json"):
            with open(os.path.join(logging_dir, x), 'r') as f:
                logs.append(json.load(f))
    return logs

logs = load_logs(logging_dir)

def extract_metrics(logs):
    steps = []
    train_loss = []
    eval_loss = []

    for log in logs:
        if 'train' in log:
            steps.append(log['step'])
            train_loss.append(log['train']['loss'])
        if 'eval' in log:
            eval_loss.append(log['eval']['eval_loss'])

    return steps, train_loss, eval_loss

steps, train_loss, eval_loss = extract_metrics(logs)

plt.figure(figsize=(12, 6))

plt.plot(steps, train_loss, label='Training Loss', color='blue')
plt.plot(steps, eval_loss, label='Validation Loss', color='orange')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png') 
plt.show()
