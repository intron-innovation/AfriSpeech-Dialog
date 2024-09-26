import re
import os
import argparse
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False)
    
    args = parser.parse_args()
    args.model_name = args.pretrained_model_path.split("/")[-1]

    return args

def extract_text_transcript(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        
    cleaned_content = re.sub(r'\d{1,2}:\d{2}\n\w+:\s', '', content)
    cleaned_content = ' '.join(cleaned_content.split())
    return cleaned_content

def prepare_data(data_dir):
    audio_paths = []
    texts = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                text_file = os.path.splitext(file)[0] + '.txt'
                text_path = os.path.join(root, text_file)
                
                # Check if the corresponding .txt file exists
                if os.path.exists(text_path):
                    transcript = extract_text_transcript(text_path)
                    
                    audio_paths.append(audio_path)
                    texts.append(transcript)

    df = pd.DataFrame({
        'audio_path': audio_paths,
        'reference': texts
    })

    return df

def write_results(data, args, score):
    model_dir = args.model_name.replace("/", "_")
    os.makedirs(f"results/{model_dir}", exist_ok=True)
    results_fname = f"results/{model_dir}/{model_dir}-wer_{score:.4f}_{len(data)}.csv"
    data.to_csv(results_fname, index=False)
    logger.info(f"Results saved to: {results_fname}")
    return results_fname