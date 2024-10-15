# AfriSpeech-Dialog

## Project Overview

**AfriSpeech-Dialog** is a project to evaluate Automatic Speech Recognition (ASR), speaker diarization, and multi-agent summarization on medical and non-medical African-accented conversations (long-form), which include code-switching.

### Main Contributions:
The project and associated paper make the following contributions:
- Introduces a dataset of ~50 simulated medical/non-medical conversations with African accents.
- Evaluates state-of-the-art (SOTA) speaker diarization models on accented speech.
- Compares the performance of open multilingual ASR models (e.g., Whisper, Conformer, MMS, XLS-R) on long-form accented speech, benchmarked against datasets from other continents.
- Evaluates multi-agent summarization of medical/non-medical conversation transcripts.

---

## Installation

To set up the environment, follow these steps:

1. Create a conda environment with Python 3.10:
    ```bash
    conda create -n afrispeech_dialog python=3.10
    ```

2. Activate the environment:
    ```bash
    conda activate afrispeech_dialog
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Data Setup

The `data/` directory contains audio samples referenced in `afrispeech_dialog_v1_47.csv`. This CSV file includes the following columns:
- `path`: The relative path to the audio files.
- `transcript`: The text transcript of the conversations.
- We have provided additional columns for demographic details about the speakers.


## Running Experiments

### ASR Experiments

1. **Model Subclassing**:
    - To benchmark a specific model, subclass the `Model` class to create a custom class for that model. You can use `src/models/whisper.py` as a guide.
    - Once you've created the class, modify `bin/main_predictions.py` to include a condition for running your custom class (around line 28).

2. **Main Script**:
    - The main entry point for running experiments is `bin/main_predictions.py`. This script handles data preprocessing (for ASR, diarization, or summarization) and writes the results to the `results/` directory.

3. **Run the Experiments**:
    - Add the project directory to your `PYTHONPATH`:
      ```bash
      export PYTHONPATH="/path/to/project"
      ```
    - To run the experiments, execute the following command:
      ```bash
      bash scripts/run.sh
      ```

### Arguments for `run.sh`:
- `data_path`: Path to the CSV file containing the dataset.
- `task`: The type of experiment to perform (`asr`, `diarization`, or `summarisation`).
- `batch_size` (optional): Batch size for processing the data.

---

## Results

Results will be saved in the `results/` directory. For ASR experiments, the result CSVs follow the format: `{model_name}/{model_name}_{task}-wer_{WER}_{len(data)}.csv`. For diarization experiments, the result CSV follow the format: `{model_name}_diarization_der_{DER}_{len(diarized_data)}.csv`.

### Reported Results
#### ASR

| Model                                      | WER (%) | Medical WER (%) | Non-Medical WER (%) |
|--------------------------------------------|---------|-----------------|---------------------|
| openai/whisper-medium                      | 21.27   | 26.49           | 19.47               |
| openai/whisper-large-v2                    | 20.82   | 23.74           | 19.81               |
| openai/whisper-large-v3                    | 20.38   | 23.81           | 19.19               |
| openai/whisper-large-v3-turbo              | 21.93   | 25.58           | 20.67               |
| distil-whisper/distil-large-v2             | 25.38   | 30.43           | 23.63               |
| distil-whisper/distil-large-v3             | 21.20   | 25.67           | 19.58               |
| Crystalcareai/Whisper-Medicalv1            | 21.21   | 25.76           | 19.63               |
| Na0s/Medical-Whisper-Large-v3              | 30.51   | 34.90           | 28.97               |
| nvidia/parakeet-rnnt-1.1b                  | 28.16   | 34.03           | 26.13               |
| nvidia/parakeet-ctct-1.1b                  | 28.97   | 34.16           | 27.19               |
| nvidia/parakeet-tdt-1.1b                   | 28.69   | 33.57           | 27.01               |
| nvidia/canary-1b                           | 22.82   | 27.40           | 21.25               |
| nvidia/stt_en_conformer_ctc_large          | 35.92   | 41.98           | 33.84               |
| nvidia/stt_en_fastconformer_transducer_large | 37.69   | 45.36           | 35.05               |
| nvidia/parakeet-tdt_ctc-110m               | 21.89   | 27.71           | 19.89               |

#### Diarization
| Model                                      | DER (%) | 
|--------------------------------------------|---------|
| deepgram-nova                              | 14.21  |
| pyannote/speaker-diarization-3.1           | 21.30  |
| soniox                                     | 20.05  |
| Revai/reverb-diarization-v2                | 26.87  |
| AssemblyAI                                 | 12.72  | 


### Summarization
| Model                                      | BERT_SOCRE(F1) (%) | LLM EVAL (%)|
|--------------------------------------------|---------|-----------|
| Gpt4o                            |90.57  |73.83|
| Gpt3.5-turbo                      | 89.99  |66.83|
| Claude_Sonnet    |88.30|  72.17|
| google/gemma-2-9b |84.68 | 73.00 |
| m42-health/Llama3-Med42-8B    |90.46|  69.67|
| Meta-Llama-3.1-8B-Instruct    |85.52| 69.83|
| meta-llama/Llama-3.2-3B-Instruct |84.92|64.83|
| OpenMeditron-Meditron3-8B     | 89.83|60.50|
| microsoft/Phi-3-mini-4k-instruct    |88.00|51.67|

                                       
