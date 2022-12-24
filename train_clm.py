import os
import math
import logging
from itertools import chain
from argparse import Namespace
import evaluate
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.testing_utils import CaptureLogger
logger = logging.getLogger(__name__)



def main(args):
    data_args = Namespace(
        model_name_or_path=args.model_name_or_path,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
        ignore_pad_token_for_loss=True,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        output_dir=args.output_dir,
        num_beams=args.num_beams,
        block_size=args.block_size,
    )

    training_args = TrainingArguments(
        output_dir=data_args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        seed=args.seed,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_accuracy",
        greater_is_better=True, # smaller eval loss is better
        save_total_limit=2, # save 2 checkpoints (best and last)
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
     # Set seed before initializing model.
    set_seed(training_args.seed)

    # Blended Skill Talk
    bst_dataset = load_dataset("blended_skill_talk")
    column_names = bst_dataset['train'].column_names

    # Tokenizer and model
    config = AutoConfig.from_pretrained(data_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(data_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(data_args.model_name_or_path, config=config)

    max_length = data_args.max_length
    padding = "max_length" if data_args.pad_to_max_length else False
    print("max length: {}, model max length: {}".format(max_length, tokenizer.model_max_length))

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Add special tokens
    # Define new special tokens: <PS>, <CTX>, <SEP>
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokenizer.add_special_tokens({'mask_token': '<MASK>'})
    tokenizer.add_tokens(['<PS>'], special_tokens=True) ## this line is updated
    tokenizer.add_tokens(['<CTX>'], special_tokens=True) ## this line is updated
    tokenizer.add_tokens(['<SEP>'], special_tokens=True) ## this line is updated
    model.resize_token_embeddings(len(tokenizer))

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    # Data processing
    def preprocess_bst(examples):
        num_entries = len(examples["free_messages"])
        persona_pieces = [
            f"<PS> {examples['personas'][0]}",
            f"<PS> {examples['personas'][1]}",
        ]
        if examples['context'] == "wizard_of_wikipedia":
            additional_context_pieces = [f"<CTX> {examples['additional_context']}. <SEP> "]
        else:
            additional_context_pieces = ["<SEP> "]

        previous_utterance_pieces = examples["previous_utterance"]
        for entry_idx in range(num_entries):
            free_message = examples['free_messages'][entry_idx]
            guided_message = examples['guided_messages'][entry_idx]

            previous_utterance = ' <SEP> '.join(previous_utterance_pieces)
            original_context = ' '.join(
                persona_pieces + additional_context_pieces
            ) + previous_utterance
            text = ' <SEP> '.join([original_context, free_message, guided_message])
            previous_utterance_pieces += [
                free_message,
                guided_message,
            ]

        with CaptureLogger(tok_logger) as cl:
            inputs = tokenizer([text], max_length=max_length, padding=padding, truncation=True)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return inputs


    def group_texts(examples):
        # ['input_ids', 'attention_mask', 'labels']
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        # total_length = len(concatenated_examples[list(examples.keys())[0]])
        # # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # # customize this part to your needs.
        # if total_length >= block_size:
        #     total_length = (total_length // block_size) * block_size
        # # Split by chunks of max_len.
        # result = {
        #     k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        #     for k, t in concatenated_examples.items()
        # }
        concatenated_examples["labels"] = concatenated_examples["input_ids"].copy()
        return concatenated_examples

    # Tokenize train, eval, test dataset
    if training_args.do_train:
        train_dataset = bst_dataset['train']
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_bst,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        train_dataset = train_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        print("train dataset: ", train_dataset)
    
    if training_args.do_eval:
        eval_dataset = bst_dataset['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)


        eval_dataset = eval_dataset.map(
            preprocess_bst,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        eval_dataset = eval_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        print("validation dataset: ", eval_dataset)

    if training_args.do_predict:
        predict_dataset = bst_dataset['test']
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_bst,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        predict_dataset = predict_dataset.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        print("test dataset: ", predict_dataset)

    # Data collator
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    data_collator = default_data_collator

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--model_name_or_path', 
                        type=str, 
                        default='microsoft/DialoGPT-small', 
                        choices=['microsoft/DialoGPT-small', 'gpt2'],
                        help='The model checkpoint for weights initialization.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='results/dialogpt',
                        help='The output directory where the model predictions and checkpoints will be written.')
    parser.add_argument('--num_train_epochs',
                        type=int,
                        default=50,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--per_device_train_batch_size',
                        type=int,
                        default=10,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_device_eval_batch_size',
                        type=int,
                        default=20,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=20,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--do_train',
                        action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval',
                        action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_predict',
                        action='store_true',
                        help='Whether to run predictions on the test set.')
    parser.add_argument('--max_train_samples',
                        type=int,
                        default=None,
                        help='For debugging purposes or quicker training, truncate the number of training examples to this.')
    parser.add_argument('--max_eval_samples',
                        type=int,
                        default=None,
                        help='For debugging purposes or quicker training, truncate the number of evaluation examples to this.')
    parser.add_argument('--max_predict_samples',
                        type=int,
                        default=None,
                        help='For debugging purposes or quicker training, truncate the number of prediction examples to this.')
    parser.add_argument('--max_length',
                        type=int,
                        default=512,
                        help='The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--pad_to_max_length',
                        action='store_true',
                        help='Whether to pad all samples to model maximum sentence length.')
    parser.add_argument('--block_size',
                        type=int,
                        default=None,
                        help='Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training.')
    parser.add_argument('--num_beams',
                        type=int,
                        default=4,
                        help='Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.')
    parser.add_argument('--overwrite_cache',
                        action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--preprocessing_num_workers',
                        type=int,
                        default=None,
                        help='The number of processes to use for the preprocessing.')

    args = parser.parse_args()

    main(args)
