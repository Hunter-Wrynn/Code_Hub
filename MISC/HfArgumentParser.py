from dataclasses import dataclass
from transformers import HfArgumentParser

@dataclass
class ModelArguments:
    model_name_or_path: str = "bert-base-uncased"

@dataclass
class DataTrainingArguments:
    max_seq_length: int = 128

@dataclass
class TrainingArguments:
    learning_rate: float = 2e-5

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

print(model_args.model_name_or_path)
print(data_args.max_seq_length)
print(training_args.learning_rate)
