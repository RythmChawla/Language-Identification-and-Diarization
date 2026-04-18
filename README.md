# Language Identification and Diarization

A complete pipeline for Language Identification (LID) and Speaker Diarization that processes audio recordings to determine which language is spoken and who is speaking at different points in time.

## Overview

This project analyzes audio input to perform:

* Language detection for spoken segments
* Speaker segmentation and identification
* Timestamp-based labeling of speakers and languages
* Combined multilingual and multi-speaker analysis

It can be applied in use cases such as meeting transcription, call center analytics, podcast processing, and multilingual speech systems.

## Key Concepts

### Language Identification (LID)

Language Identification involves automatically determining the language spoken in a given audio segment using extracted audio features and trained models.

### Speaker Diarization

Speaker Diarization segments audio based on speaker identity. The goal is to answer: "who spoke when?"

Typical steps include:

* Voice Activity Detection (VAD)
* Audio segmentation
* Feature or embedding extraction
* Clustering of speaker representations

## Pipeline Architecture

```
Audio Input
↓
Preprocessing (resampling, normalization, noise reduction)
↓
Voice Activity Detection
↓
Segmentation
↓
Feature Extraction (MFCC / embeddings)
↓
Parallel Processing:
   ├── Language Identification Model
   └── Speaker Diarization Model
↓
Clustering and Label Assignment
↓
Final Output (speaker + language with timestamps)
```

## Features

* Supports raw audio input
* Multilingual language detection
* Multi-speaker diarization
* Segment-wise analysis
* Structured output with timestamps

## Tech Stack

* Python
* Librosa or PyAudio for audio processing
* Scikit-learn for clustering
* NumPy and Pandas
* Machine learning or deep learning models for LID and embeddings

## Project Structure

```
├── DD/                # Diarization files
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── diarization.py
│   ├── language_id.py
│   └── pipeline.py
├── labels/              # Dataset without audio files(can download the merlion ccs challenge dataset)
├── README     
├── fine_tuning.py.txt   # for Language Identification
└── running model.py     # full pipeline
```

## Installation

```
git clone https://github.com/RythmChawla/Language-Identification-and-Diarization.git
cd Language-Identification-and-Diarization
pip install -r requirements.txt
```

## Usage

```
python src/pipeline.py --input path/to/audio.wav
```

## Output Example

```
[00:00 - 00:08] Speaker 1 | English
[00:08 - 00:15] Speaker 2 | Hindi
[00:15 - 00:25] Speaker 1 | English
```
