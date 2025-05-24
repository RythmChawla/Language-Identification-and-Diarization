import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F
from pydub import AudioSegment
from IPython.display import Audio, display

# Create results directory
results_dir = "fine_tune_Results"
os.makedirs(results_dir, exist_ok=True)

# Modified paths for output files
chinese_output_csv = os.path.join(results_dir, "chinese_audio_metadata.csv")
english_output_csv = os.path.join(results_dir, "english_audio_metadata.csv")

# Paths
audio_dir_path = "/home/teaching/Desktop/priyam/_audio"
label_dir_path = "/home/teaching/Desktop/priyam/labels"
labels_dir = "/home/teaching/Desktop/priyam/labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv"

# Load data
df = pd.read_csv(labels_dir)
df["start"] = df["start"].astype(float)
df["end"] = df["end"].astype(float)

print(f"Before shape", df.shape)
df["label"] = df["language_tag"].apply(lambda x: 1 if x == "English" else 0)
df["dev_eval_encoded"] = df["dev_eval_status"].apply(lambda x: 1 if x == "dev" else 0)
print(f"After shape", df.shape)

print(f"Total number of audios are: {df.shape[0]}")
print("Total number of english segments:", df["label"].sum())
print("Total number of chinese segments:", df.shape[0] - df["label"].sum())

# Prepare metadata CSVs
chinese_rows = []
english_rows = []
limit = 650
chinese_count = 0
english_count = 0
min_duration_ms = 3000

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

# Load the saved CSVs
df_chinese = pd.read_csv(chinese_output_csv)
df_english = pd.read_csv(english_output_csv)
print("Number of chinese audio ", len(df_chinese))
print("Number of english audio ", len(df_english))

# Define the dataset class
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

from transformers import HubertModel, HubertConfig
import torch.nn as nn
import torch

class HubertClassifierWithFreeze(nn.Module):
    def __init__(self, hubert_model_name="facebook/hubert-large-ls960-ft", num_labels=2):
        super().__init__()
        self.base_model = HubertModel.from_pretrained(hubert_model_name)
        self.config = self.base_model.config

        hidden_size = self.config.hidden_size  # 1024 for large model

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_labels)
        )

        # Freeze the base HuBERT model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_values, attention_mask=None):
        outputs = self.base_model(input_values=input_values, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # shape: [batch, time, hidden]
        
        # Pooling (mean over time dimension)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)
        return logits


# Training functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        attention_mask = (inputs != 0).long()
        
        # Adapt based on model type
        if isinstance(model, torch.nn.Module) and not isinstance(model, Wav2Vec2ForSequenceClassification):
            # For custom model
            logits = model(input_values=inputs, attention_mask=attention_mask)
            loss = criterion(logits, labels)
        else:
            # For huggingface model
            outputs = model(input_values=inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            attention_mask = (inputs != 0).long()

            # Adapt based on model type
            if isinstance(model, torch.nn.Module) and not isinstance(model, Wav2Vec2ForSequenceClassification):
                # For custom model
                logits = model(input_values=inputs, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            else:
                # For huggingface model
                outputs = model(input_values=inputs, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                logits = outputs.logits

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy

def get_logits_and_labels(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Getting logits"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_mask = (inputs != 0).long()

            # Adapt based on model type
            if isinstance(model, torch.nn.Module) and not isinstance(model, Wav2Vec2ForSequenceClassification):
                # For custom model
                logits = model(input_values=inputs, attention_mask=attention_mask)
            else:
                # For huggingface model
                outputs = model(input_values=inputs, attention_mask=attention_mask)
                logits = outputs.logits

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of class "1" (English)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_probs), np.array(all_labels)

# ROC and EER functions
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

def compute_eer_from_roc(fpr, tpr, y_score):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    eer_index = np.argmin(abs_diffs)
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = y_score[eer_index]
    return eer, eer_threshold

def plot_roc_manual(y_true, y_score, title="ROC Curve", save_path="roc_curve.png"):
    fpr, tpr = compute_roc(y_true, y_score)
    auc_score = compute_auc(fpr, tpr)
    eer, eer_threshold = compute_eer_from_roc(fpr, tpr, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{title}\nEER = {eer:.4f}, Threshold = {eer_threshold:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, save_path))
    plt.show()

    print(f"✅ AUC: {auc_score:.4f}")
    print(f"🔍 EER: {eer:.4f}")
    print(f"🎯 Optimal Threshold: {eer_threshold:.4f}")
    
    return auc_score, eer, eer_threshold

# Main training code
if __name__ == "__main__":
    # Load metadata
    df_chinese = pd.read_csv(chinese_output_csv)
    df_english = pd.read_csv(english_output_csv)
    df = pd.concat([df_chinese, df_english]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    # Feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Datasets and loaders
    train_dataset = LanguageIDDataset(train_df, feature_extractor)
    val_dataset = LanguageIDDataset(val_df, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create custom model with frozen base - CORRECTED LINE
    custom_model = HubertClassifierWithFreeze(hubert_model_name="facebook/hubert-large-ls960-ft", num_labels=2).to(device)
    
    # First phase: Train only the classifier head
    print("\n\n🔬 PHASE 1: FEATURE EXTRACTION - TRAINING CLASSIFIER HEAD ONLY 🔬\n")
    
    # Setup optimizer and criterion for phase 1
    optimizer_phase1 = torch.optim.AdamW(custom_model.classifier.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop - Phase 1 (Feature Extraction)
    phase1_epochs = 4
    phase1_train_losses = []
    phase1_val_losses = []
    phase1_train_accuracies = []
    phase1_val_accuracies = []
    
    for epoch in range(phase1_epochs):
        print(f"\nPhase 1 - Epoch {epoch + 1}/{phase1_epochs}")
        
        train_loss, train_acc = train_one_epoch(custom_model, train_loader, optimizer_phase1, criterion, device)
        val_loss, val_acc = validate(custom_model, val_loader, criterion, device)
        
        phase1_train_losses.append(train_loss)
        phase1_val_losses.append(val_loss)
        phase1_train_accuracies.append(train_acc)
        phase1_val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    
    # Save phase 1 model
    torch.save(custom_model.state_dict(), os.path.join(results_dir, "phase1_feature_extraction_model.pt"))
    
    # Plot Phase 1 performance
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(phase1_train_accuracies, label="Train Accuracy")
    plt.plot(phase1_val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Phase 1: Feature Extraction - Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(phase1_train_losses, label="Train Loss")
    plt.plot(phase1_val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 1: Feature Extraction - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase1_performance.png"))
    plt.show()

    # Save metrics for phase 1
    metrics_phase1 = {
        'train_losses': phase1_train_losses,
        'val_losses': phase1_val_losses,
        'train_accuracies': phase1_train_accuracies,
        'val_accuracies': phase1_val_accuracies
    }
    pd.DataFrame(metrics_phase1).to_csv(os.path.join(results_dir, 'phase1_metrics.csv'), index=False)

    # Get Phase 1 ROC
    print("\nGenerating ROC curve for Phase 1 model...")
    phase1_probs, phase1_labels = get_logits_and_labels(custom_model, val_loader, device)
    phase1_auc, phase1_eer, phase1_threshold = plot_roc_manual(
        phase1_labels, 
        phase1_probs, 
        title="Phase 1: Feature Extraction - ROC Curve", 
        save_path="phase1_roc_curve.png"
    )
    
    # Second phase: Fine-tune the entire model
    print("\n\n🔍 PHASE 2: FINE-TUNING - TRAINING ENTIRE MODEL 🔍\n")
    
    # Unfreeze the base model
    for param in custom_model.base_model.parameters():
        param.requires_grad = True
    
    # Setup optimizer for phase 2 with lower learning rate for fine-tuning
    optimizer_phase2 = torch.optim.AdamW([
        {'params': custom_model.base_model.parameters(), 'lr': 1e-5},  # Lower LR for pre-trained base
        {'params': custom_model.classifier.parameters(), 'lr': 5e-5}   # Higher LR for classifier
    ])
    
    # Training loop - Phase 2 (Fine-tuning)
    phase2_epochs = 4
    phase2_train_losses = []
    phase2_val_losses = []
    phase2_train_accuracies = []
    phase2_val_accuracies = []
    
    for epoch in range(phase2_epochs):
        print(f"\nPhase 2 - Epoch {epoch + 1}/{phase2_epochs}")
        
        train_loss, train_acc = train_one_epoch(custom_model, train_loader, optimizer_phase2, criterion, device)
        val_loss, val_acc = validate(custom_model, val_loader, criterion, device)
        
        phase2_train_losses.append(train_loss)
        phase2_val_losses.append(val_loss)
        phase2_train_accuracies.append(train_acc)
        phase2_val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    
    # Save fine-tuned model
    torch.save(custom_model.state_dict(), os.path.join(results_dir, "phase2_fine_tuned_model.pt"))
    
    # Save metrics for phase 2
    metrics_phase2 = {
        'train_losses': phase2_train_losses,
        'val_losses': phase2_val_losses,
        'train_accuracies': phase2_train_accuracies,
        'val_accuracies': phase2_val_accuracies
    }
    pd.DataFrame(metrics_phase2).to_csv(os.path.join(results_dir, 'phase2_metrics.csv'), index=False)
    
    # Plot Phase 2 performance
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(phase2_train_accuracies, label="Train Accuracy")
    plt.plot(phase2_val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Phase 2: Fine-Tuning - Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(phase2_train_losses, label="Train Loss")
    plt.plot(phase2_val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 2: Fine-Tuning - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase2_performance.png"))
    plt.show()
    
    # Get Phase 2 ROC
    print("\nGenerating ROC curve for Phase 2 model...")
    phase2_probs, phase2_labels = get_logits_and_labels(custom_model, val_loader, device)
    phase2_auc, phase2_eer, phase2_threshold = plot_roc_manual(
        phase2_labels, 
        phase2_probs, 
        title="Phase 2: Fine-Tuning - ROC Curve", 
        save_path="phase2_roc_curve.png"
    )
    # Save ROC data
    roc_results = {
        'phase1_auc': phase1_auc,
        'phase1_eer': phase1_eer,
        'phase1_threshold': phase1_threshold,
        'phase2_auc': phase2_auc,
        'phase2_eer': phase2_eer,
        'phase2_threshold': phase2_threshold,
        'auc_improvement': phase2_auc - phase1_auc,
        'eer_improvement': phase1_eer - phase2_eer
    }
    pd.DataFrame([roc_results]).to_csv(os.path.join(results_dir, 'roc_metrics.csv'), index=False)
    
    # Plot combined training progress
    plt.figure(figsize=(15, 10))
    
    # Combined Accuracy Plot
    plt.subplot(2, 2, 1)
    plt.plot(phase1_train_accuracies, label="Phase 1 Train Accuracy", marker='o')
    plt.plot(phase1_val_accuracies, label="Phase 1 Val Accuracy", marker='o')
    plt.plot(range(phase1_epochs, phase1_epochs + phase2_epochs), 
             phase2_train_accuracies, 
             label="Phase 2 Train Accuracy", 
             marker='s')
    plt.plot(range(phase1_epochs, phase1_epochs + phase2_epochs), 
             phase2_val_accuracies, 
             label="Phase 2 Val Accuracy", 
             marker='s')
    plt.axvline(x=phase1_epochs-0.5, color='r', linestyle='--', 
                label="Feature Extraction → Fine-Tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy: Feature Extraction vs Fine-Tuning")
    plt.legend()
    plt.grid(True)
    
    # Combined Loss Plot
    plt.subplot(2, 2, 2)
    plt.plot(phase1_train_losses, label="Phase 1 Train Loss", marker='o')
    plt.plot(phase1_val_losses, label="Phase 1 Val Loss", marker='o')
    plt.plot(range(phase1_epochs, phase1_epochs + phase2_epochs), 
             phase2_train_losses, 
             label="Phase 2 Train Loss", 
             marker='s')
    plt.plot(range(phase1_epochs, phase1_epochs + phase2_epochs), 
             phase2_val_losses, 
             label="Phase 2 Val Loss", 
             marker='s')
    plt.axvline(x=phase1_epochs-0.5, color='r', linestyle='--',
                label="Feature Extraction → Fine-Tuning")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss: Feature Extraction vs Fine-Tuning")
    plt.legend()
    plt.grid(True)
    
    # Comparative ROC Curves
    plt.subplot(2, 1, 2)
    # Calculate ROCs
    fpr1, tpr1 = compute_roc(phase1_labels, phase1_probs)
    fpr2, tpr2 = compute_roc(phase2_labels, phase2_probs)
    
    plt.plot(fpr1, tpr1, label=f"Phase 1: Feature Extraction (AUC = {phase1_auc:.4f}, EER = {phase1_eer:.4f})")
    plt.plot(fpr2, tpr2, label=f"Phase 2: Fine-Tuning (AUC = {phase2_auc:.4f}, EER = {phase2_eer:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison: Feature Extraction vs Fine-Tuning")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combined_training_comparison.png"))
    plt.show()
    
    # Save validation probabilities and labels for further analysis
    validation_results = {
        'phase1_probs': phase1_probs,
        'phase1_labels': phase1_labels,
        'phase2_probs': phase2_probs,
        'phase2_labels': phase2_labels
    }
    np.savez(os.path.join(results_dir, 'validation_probabilities.npz'), **validation_results)
    
    # Print final results
    print("\n\n✅ TRAINING COMPLETE! ✅")
    print(f"Phase 1 (Feature Extraction) Final Validation Accuracy: {phase1_val_accuracies[-1]:.4f}")
    print(f"Phase 2 (Fine-Tuning) Final Validation Accuracy: {phase2_val_accuracies[-1]:.4f}")
    print(f"Improvement from Fine-Tuning: {phase2_val_accuracies[-1] - phase1_val_accuracies[-1]:.4f}")
    
    print(f"\nPhase 1 AUC: {phase1_auc:.4f}, EER: {phase1_eer:.4f}")
    print(f"Phase 2 AUC: {phase2_auc:.4f}, EER: {phase2_eer:.4f}")
    print(f"AUC Improvement: {phase2_auc - phase1_auc:.4f}")
    print(f"EER Improvement: {phase1_eer - phase2_eer:.4f} (lower is better)")
    
    # Save a summary of results
   
    with open(os.path.join(results_dir, 'results_summary.txt'), 'w') as f:
        f.write("LANGUAGE ID MODEL - TRAINING RESULTS\n")
        f.write("==================================\n\n")
        f.write(f"Phase 1 (Feature Extraction) Final Validation Accuracy: {phase1_val_accuracies[-1]:.4f}\n")
        f.write(f"Phase 2 (Fine-Tuning) Final Validation Accuracy: {phase2_val_accuracies[-1]:.4f}\n")
        f.write(f"Improvement from Fine-Tuning: {phase2_val_accuracies[-1] - phase1_val_accuracies[-1]:.4f}\n\n")
        f.write(f"Phase 1 AUC: {phase1_auc:.4f}, EER: {phase1_eer:.4f}, Threshold: {phase1_threshold:.4f}\n")
        f.write(f"Phase 2 AUC: {phase2_auc:.4f}, EER: {phase2_eer:.4f}, Threshold: {phase2_threshold:.4f}\n")
        f.write(f"AUC Improvement: {phase2_auc - phase1_auc:.4f}\n")
        f.write(f"EER Improvement: {phase1_eer - phase2_eer:.4f} (lower is better)\n")