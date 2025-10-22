from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DistilBertTokenizer
)
import torch
import evaluate




############ Check Device ###########
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")
#####################################


############ Load dataset ###########
print("\nLoading IMDB dataset...")
dataset = load_dataset("imdb")
#####################################

############ Load tokenizer ###########
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

print(" Tokenizing data...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

############ Load Model ###########
print("\nLoading DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
).to(device)
#####################################

############ Define Metric ###########
print("📏 Loading evaluation metric...")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

#####################################


############ Device dependent settings ###########
if device.type == "mps":
    batch_size = 8
    fp16 = False
elif device.type == "cuda":
    batch_size = 16
    fp16 = True
else:
    batch_size = 4
    fp16 = False
##################################################


############ Training Arguments ###########
training_args = TrainingArguments(
    output_dir="./distilbert-imdb",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=fp16,  # only used if CUDA available
    report_to="none",  # disables wandb/tensorboard logs
)
###########################################


############ Trainer setup ###########
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
######################################

############ Fine-tune ###########
print("\nStarting fine-tuning...")
trainer.train()
##################################


############ Evaluate ###########
print("\nEvaluating model...")
results = trainer.evaluate()
print(results)
#################################


############ Saving Model ###########
print("\n💾 Saving fine-tuned model...")
trainer.save_model("./distilbert-imdb-final")
#####################################

print("\nFine-tuning complete! Model saved at './distilbert-imdb-final'")