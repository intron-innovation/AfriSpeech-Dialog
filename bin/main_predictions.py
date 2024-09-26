import os
import pandas as pd
import logging
import gc
import torch
from src.models.whisper import Whisper
from src.utils.utils import parse_arguments, prepare_asr_data, write_results
from src.utils.text_processing import post_process_preds
from transformers import set_seed


logger = logging.getLogger(__name__)
set_seed(42)

def main():
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()

    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"Loading model from {args.pretrained_model_path}")
    # please define your own model class
    if "whisper" in args.pretrained_model_path:
        model = Whisper(args.pretrained_model_path)
    else:
        raise NotImplementedError(
            f"No model class defined for {args.pretrained_model_path}"
        )
    logger.info("Model loaded successfully")

    logger.info(f"Preprocessing data at {args.data_path}")
    
    if args.task == 'asr':
        data = prepare_asr_data(args.data_dir)    

        transcripts = model.transcribe(list(data['audio_path']), batch_size=args.batch_size)
        data['hypothesis'] = transcripts

        all_wer = post_process_preds(data)

        write_results(args=args, data=data, score=all_wer)


if __name__ == "__main__":
    main()