import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from accelerate import Accelerator
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AdamWeightDecay, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Disable tqdm warning message
warnings.filterwarnings("ignore", category=UserWarning)

# Load the data
df = pd.read_csv('/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/data/processed/flavivirus.csv')

# Create classification column
def create_classification_column(df):
    df['classification'] = df.apply(lambda row: {'Host': row['Y2_subVector'], 'Sequence': row['X1_Nucleotide']}, axis=1)
    df = df.drop(['X1_Nucleotide', 'Y2_subVector'], axis=1)
    return df

df = create_classification_column(df)

# Convert to Hugging Face dataset
hf = Dataset.from_pandas(df)

# Split dataset into train, validation, and test
hf = hf.train_test_split(train_size=0.8, seed=42)
hf_clean = hf["train"].train_test_split(train_size=0.8, seed=42)
hf_clean["validation"] = hf_clean.pop("test")
hf_clean["test"] = hf["test"]
hf = hf_clean

print("\n\nDataset after Breaking into Train_Val_Test")
print(hf)

# Initialize Accelerator
accelerator = Accelerator()

# Tokenizer and model checkpoint
tokenizer = AutoTokenizer.from_pretrained('/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/models/virusTrained100k')
source_lang = "Sequence"
target_lang = "Host"
prefix = "Sequence Host:"

# Preprocessing function
def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["classification"]]
    targets = [example[target_lang] for example in examples["classification"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=1737, truncation=True)
    return model_inputs

# Tokenize datasets
tokenized_hf = hf.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/models/virusTrained100k")

# Define optimizer
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

# Initialize model
model = AutoModelForSeq2SeqLM.from_pretrained("/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/models/virusTrained100k", from_flax=True)

# Accelerator prepares the model and optimizer
model, optimizer = accelerator.prepare(model, optimizer)

# Evaluation metric
metric = evaluate.load("accuracy")

# Confusion matrix and metric computation function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    epoch = trainer.state.epoch
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    combined_labels = decoded_preds + decoded_labels
    le = LabelEncoder()

    try:
        le.fit(combined_labels)
    except ValueError:
        le.fit([label for label in combined_labels if label not in le.classes_])

    numeric_preds = le.transform(decoded_preds)
    numeric_labels = le.transform(decoded_labels)

    cm = confusion_matrix(numeric_labels, numeric_preds)
    plt.figure(figsize=(36, 24))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix Host - Epoch {epoch}")
    plt.savefig(f"confusion_matrix_{epoch}.png")

    accuracy = metric.compute(predictions=numeric_preds, references=numeric_labels)
    return accuracy

# Training arguments and trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="host_classification_flavi",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=15,
    predict_with_generate=True,
    fp16=True  # Mixed precision enabled
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_hf['train'],
    eval_dataset=tokenized_hf['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Evaluate on test set
results = trainer.evaluate(eval_dataset=tokenized_hf['test'])

print("\n\n RESULTS OF TEST DATASET !!")
print(results)

# Extract and save training history
train_metrics = trainer.state.log_history
train_df = pd.DataFrame(train_metrics)
train_df.to_csv("host_15_epochs_metrics_flavi.csv", index=False)
