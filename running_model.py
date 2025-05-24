import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
audio_dir_path="/home/teaching/Desktop/priyam/_audio"
for path, _ , files in os.walk(audio_dir_path):
    print(path ,len(_),len(files))

label_dir_path="/home/teaching/Desktop/priyam/labels"
print(os.listdir(label_dir_path))

labels_dir= "/home/teaching/Desktop/priyam/labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv"
df=pd.read_csv(labels_dir)

print(df.head(3))

df["start"]=df["start"].astype(float)
df["end"]=df["end"].astype(float)

print(f"Before shape",df.shape)
df["label"] = df["language_tag"].apply(lambda x: 1 if x == "English" else 0)
df["dev_eval_encoded"] = df["dev_eval_status"].apply(lambda x: 1 if x == "dev" else 0)


print(f"After shape",df.shape)

print(f"Total number of audios are :{df.shape[0]}")
print("Total number of english segments:",df["label"].sum())
print("Total number of chinese segments:",df.shape[0]-df["label"].sum())


print("--------------------------------------------\n\n\n")

number_of_chinese_audio = 0
number_of_english_audio=0

for i in range(len(df)):
    audio_name = df.iloc[i]["audio_name"]
    audio_path = os.path.join(audio_dir_path, audio_name)
    print(audio_path)
    print(df.iloc[i]["start"] , df.iloc[i]["end"] ,df.iloc[i]["label"],df.iloc[i]["length"] ,df.iloc[i]["dev_eval_encoded"])
    break


import os
import pandas as pd

# Paths
chinese_output_csv = "chinese_audio_metadata.csv"
english_output_csv = "english_audio_metadata.csv"

# Counters and containers
chinese_rows = []
english_rows = []
limit = 650
chinese_count = 0
english_count = 0

min_duration_ms=3000


for i in range(len(df)):
    row = df.iloc[i]
    label = row["label"]
    audio_name = row["audio_name"]
    audio_path = os.path.join(audio_dir_path, audio_name)
    start = row["start"]
    end = row["end"]
    duration = end - start

    # Skip if too short
    if duration < min_duration_ms:
        continue

    if not os.path.exists(audio_path):
        continue

    row_data = row.to_dict()
    row_data["full_audio_path"] = audio_path

    if label == 0 and chinese_count < limit:
        chinese_rows.append(row_data)
        chinese_count += 1

    elif label == 1 and english_count < limit:
        english_rows.append(row_data)
        english_count += 1
    
    if chinese_count >= limit and english_count >= limit:
        break


# Save to CSV
pd.DataFrame(chinese_rows).to_csv(chinese_output_csv, index=False)
pd.DataFrame(english_rows).to_csv(english_output_csv, index=False)

print(f"\n\n✅ CSVs saved with {len(chinese_rows)} Chinese and {len(english_rows)} English audio paths and metadata.✅\n\n")



chinise_dir ="/home/teaching/Desktop/chinese_audio_metadata.csv"
english_dir ="/home/teaching/Desktop/english_audio_metadata.csv"

chinese_df = pd.read_csv(chinise_dir)
english_df = pd.read_csv(english_dir)
print("Number of chinese audio ",len(chinese_df))
print("Number of chinese audio ",len(english_df))


print(f"\n\n\n\nSEPERATED CHINESE DATAFRAME 🛠️")
print(chinese_df.head())


print(f"\n\n\n\nSEPERATED ENGLISH DATAFRAME 🛠️")
print(english_df.head())

from pydub import AudioSegment
from IPython.display import Audio, display
import pandas as pd
import os


df_chinese = pd.read_csv("chinese_audio_metadata.csv")


from pydub import AudioSegment
from IPython.display import Audio, display
import pandas as pd
import os


df_chinese = pd.read_csv("chinese_audio_metadata.csv")


for i in range(600,601):
    row = df_chinese.iloc[i]
    audio_path = row["full_audio_path"]
    start_ms = int(row["start"])
    end_ms = int(row["end"])

    print(f"▶️ Playing: {audio_path}")
    print(f"⏱️ Segment: {start_ms}ms to {end_ms}ms")


    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_ms:end_ms]


    clip_path = f"chinese_clip_{i+1}.wav"
    segment.export(clip_path, format="wav")

    display(Audio(clip_path))




df_english = pd.read_csv("english_audio_metadata.csv")


for i in range(1):
    row = df_english.iloc[i]
    audio_path = row["full_audio_path"]
    start_ms = int(row["start"])
    end_ms = int(row["end"])

    print(f"▶️ Playing: {audio_path}")
    print(f"⏱️ Segment: {start_ms}ms to {end_ms}ms")

    audio = AudioSegment.from_file(audio_path)
    segment = audio[start_ms:end_ms]

    clip_path = f"english_clip_{i+1}.wav"
    segment.export(clip_path, format="wav")
    display(Audio(clip_path))




print(f"\n\n\n\n\nWORKING ON MODEL\n\n\n\n\n\n")

import os
import torch
import torchaudio
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments
)

class LanguageIDDataset(Dataset):
    def __init__(self, df, feature_extractor, target_len_ms=3500):
        self.df = df
        self.feature_extractor = feature_extractor
        self.target_len = int(16000 * target_len_ms / 1000)  # samples for 3.5s

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["full_audio_path"]
        label = row["label"]

        start_ms = row["start"]
        end_ms = row["end"]

        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # mono
        
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

        # Convert start/end to sample indices
        start_sample = int((start_ms / 1000) * 16000)
        end_sample = int((end_ms / 1000) * 16000)

        segment = waveform[start_sample:end_sample]

        # Ensure fixed length: trim or pad to exactly 3.5s
        if segment.shape[0] > self.target_len:
            segment = segment[:self.target_len]
        elif segment.shape[0] < self.target_len:
            pad_len = self.target_len - segment.shape[0]
            segment = torch.nn.functional.pad(segment, (0, pad_len))

        # Feature extraction
        inputs = self.feature_extractor(
            segment.numpy(), sampling_rate=16000, return_tensors="pt", padding=False
        )
        input_values = inputs["input_values"].squeeze(0)

        return input_values, torch.tensor(label, dtype=torch.long)



# Load metadata
df_chinese = pd.read_csv("chinese_audio_metadata.csv")
df_english = pd.read_csv("english_audio_metadata.csv")
df = pd.concat([df_chinese, df_english]).sample(frac=1).reset_index(drop=True)

# Split
train_df = df.sample(frac=0.9, random_state=42)
val_df = df.drop(train_df.index)

# Feature extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Datasets and loaders
train_dataset = LanguageIDDataset(train_df, feature_extractor)
val_dataset = LanguageIDDataset(val_df, feature_extractor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53", num_labels=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        attention_mask = (inputs != 0).long()

        outputs = model(input_values=inputs, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            attention_mask = (inputs != 0).long()

            outputs = model(input_values=inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


num_epochs = 7


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

# Save logs
np.savetxt("train_loss.txt", train_losses)
np.savetxt("val_accuracy.txt", val_accuracies)

np.savetxt("train_accuracy.txt", train_accuracies)

import matplotlib.pyplot as plt

# Accuracy Plot
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy ✅")
plt.plot(val_accuracies, label="Validation Accuracy 📊")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.show()

# Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()








import numpy as np
import matplotlib.pyplot as plt

def compute_roc(y_true, y_score):
    # Sort scores and corresponding labels
    desc_score_indices = np.argsort(-y_score)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    tpr = tps / tps[-1]  # True Positive Rate
    fpr = fps / fps[-1]  # False Positive Rate

    return fpr, tpr

def compute_auc(fpr, tpr):
    return np.trapz(tpr, fpr)

def compute_eer_from_roc(fpr, tpr):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_index = np.argmin(abs_diffs)
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    return eer

def plot_roc_manual(y_true, y_score, save_path="roc_curve_manual.png"):
    fpr, tpr = compute_roc(y_true, y_score)
    auc_score = compute_auc(fpr, tpr)
    eer = compute_eer_from_roc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"ROC Curve (EER = {eer:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

    print(f"✅ AUC: {auc_score:.4f}")
    print(f"🔍 EER: {eer:.4f}")

import torch
import torch.nn.functional as F

def get_logits_and_labels(model, dataloader):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Getting logits"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            attention_mask = (inputs != 0).long()
            outputs = model(input_values=inputs, attention_mask=attention_mask)
            logits = outputs.logits

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class "1" (English)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


# Run ROC + EER analysis
val_probs, val_labels = get_logits_and_labels(model, val_loader)
plot_roc_manual(val_labels, val_probs)
