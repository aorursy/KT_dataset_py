""" Fine-tuning the library models for question-answering."""





import logging

import os

from dataclasses import dataclass, field

from typing import Optional



from transformers import (

    AutoConfig,

    AutoTokenizer,

    HfArgumentParser,

    TFAutoModelForQuestionAnswering,

    TFTrainer,

    TFTrainingArguments,

    squad_convert_examples_to_features,

)

from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor

logger = logging.getLogger(__name__)
@dataclass

class ModelArguments:

    """

    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    """



    model_name_or_path: str = field(

        default="bert-large-uncased-whole-word-masking",

        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}

    )

    config_name: Optional[str] = field(

        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}

    )

    tokenizer_name: Optional[str] = field(

        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}

    )

    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})

    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,

    # or just modify its tokenizer_config.json.

    cache_dir: Optional[str] = field(

        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}

    )
@dataclass

class DataTrainingArguments:

    """

    Arguments pertaining to what data we are going to input our model for training and eval.

    """



    data_dir: Optional[str] = field(

        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}

    )

    max_seq_length: int = field(

        default=128,

        metadata={

            "help": "The maximum total input sequence length after tokenization. Sequences longer "

            "than this will be truncated, sequences shorter will be padded."

        },

    )

    doc_stride: int = field(

        default=128,

        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},

    )

    max_query_length: int = field(

        default=64,

        metadata={

            "help": "The maximum number of tokens for the question. Questions longer than this will "

            "be truncated to this length."

        },

    )

    max_answer_length: int = field(

        default=30,

        metadata={

            "help": "The maximum length of an answer that can be generated. This is needed because the start "

            "and end predictions are not conditioned on one another."

        },

    )

    overwrite_cache: bool = field(

        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}

    )

    version_2_with_negative: bool = field(

        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}

    )

    null_score_diff_threshold: float = field(

        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}

    )

    n_best_size: int = field(

        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}

    )

    lang_id: int = field(

        default=0,

        metadata={

            "help": "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"

        },

    )
model_args = ModelArguments(model_name_or_path='bert-large-uncased-whole-word-masking', 

                            config_name=None, 

                            tokenizer_name=None, 

                            use_fast=False, 

                            cache_dir=None)





data_args = DataTrainingArguments(data_dir=None, 

                                  max_seq_length=384, 

                                  doc_stride=128, 

                                  max_query_length=64, 

                                  max_answer_length=30, 

                                  overwrite_cache=False, 

                                  version_2_with_negative=False, 

                                  null_score_diff_threshold=0.0, 

                                  n_best_size=20, 

                                  lang_id=0)



training_args = TFTrainingArguments(output_dir='output', 

                                    overwrite_output_dir=False, 

                                    do_train=True, 

                                    do_eval=True, 

                                    do_predict=False, 

                                    evaluate_during_training=False, 

                                    per_gpu_train_batch_size=8, 

                                    per_gpu_eval_batch_size=8, 

                                    gradient_accumulation_steps=1, 

                                    learning_rate=3e-05, 

                                    weight_decay=0.0, 

                                    adam_epsilon=1e-08, 

                                    max_grad_norm=1.0, 

                                    num_train_epochs=2.0, 

                                    max_steps=-1, 

                                    warmup_steps=0, 

                                    logging_dir=None, 

                                    logging_first_step=False, 

                                    logging_steps=500, 

                                    save_steps=500, 

                                    save_total_limit=None, 

                                    no_cuda=False, 

                                    seed=42, 

                                    fp16=False, 

                                    fp16_opt_level='O1', 

                                    local_rank=-1, 

                                    tpu_num_cores=None, 

                                    tpu_metrics_debug=False, 

                                    optimizer_name='adam', 

                                    mode='text-classification', 

                                    loss_name='SparseCategoricalCrossentropy', 

                                    tpu_name=None, 

                                    end_lr=0, 

                                    eval_steps=1000, 

                                    debug=False)
if (

        os.path.exists(training_args.output_dir)

        and os.listdir(training_args.output_dir)

        and training_args.do_train

        and not training_args.overwrite_output_dir

    ):

        raise ValueError(

            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."

        )



# Setup logging

logging.basicConfig(

    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",

    datefmt="%m/%d/%Y %H:%M:%S",

    level=logging.INFO,

)

logger.info(

    "n_gpu: %s, distributed training: %s, 16-bits training: %s",

    training_args.n_gpu,

    bool(training_args.n_gpu > 1),

    training_args.fp16,

)

logger.info("Training/evaluation parameters %s", training_args)
# Prepare Question-Answering task

    # Load pretrained model and tokenizer

    #

    # Distributed training:

    # The .from_pretrained methods guarantee that only one local process can concurrently

    # download model & vocab.



config = AutoConfig.from_pretrained(

    model_args.config_name if model_args.config_name else model_args.model_name_or_path,

    cache_dir=model_args.cache_dir,

)

tokenizer = AutoTokenizer.from_pretrained(

    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,

    cache_dir=model_args.cache_dir,

    use_fast=model_args.use_fast,

)



with training_args.strategy.scope():

    model = TFAutoModelForQuestionAnswering.from_pretrained(

        model_args.model_name_or_path,

        from_pt=bool(".bin" in model_args.model_name_or_path),

        config=config,

        cache_dir=model_args.cache_dir,

    )
!pip install tensorflow_datasets
# Get datasets

if not data_args.data_dir:

    if data_args.version_2_with_negative:

        logger.warn("tensorflow_datasets does not handle version 2 of SQuAD. Switch to version 1 automatically")



    try:

        import tensorflow_datasets as tfds

    except ImportError:

        raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")



    tfds_examples = tfds.load("squad")

    train_examples = (

        SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=False)

        if training_args.do_train

        else None

    )

    eval_examples = (

        SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=True)

        if training_args.do_eval

        else None

    )

else:

    processor = SquadV2Processor() if data_args.version_2_with_negative else SquadV1Processor()

    train_examples = processor.get_train_examples(data_args.data_dir) if training_args.do_train else None

    eval_examples = processor.get_dev_examples(data_args.data_dir) if training_args.do_eval else None
train_dataset = (

    squad_convert_examples_to_features(

        examples=train_examples,

        tokenizer=tokenizer,

        max_seq_length=data_args.max_seq_length,

        doc_stride=data_args.doc_stride,

        max_query_length=data_args.max_query_length,

        is_training=True,

        return_dataset="tf",

    )

    if training_args.do_train

    else None

)
eval_dataset = (

    squad_convert_examples_to_features(

        examples=eval_examples,

        tokenizer=tokenizer,

        max_seq_length=data_args.max_seq_length,

        doc_stride=data_args.doc_stride,

        max_query_length=data_args.max_query_length,

        is_training=False,

        return_dataset="tf",

    )

    if training_args.do_eval

    else None

)
# Initialize our Trainer

trainer = TFTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,)



# Training

if training_args.do_train:

    trainer.train()

    trainer.save_model()

    tokenizer.save_pretrained(training_args.output_dir)