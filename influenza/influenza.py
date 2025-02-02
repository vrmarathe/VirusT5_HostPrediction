import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import evaluate

# Disable tqdm warning message
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv('/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/data/processed/cleaned_influenza_dataset.csv')

def create_classification_column(df):
    df['classification'] = df.apply(lambda row: {'Host': row['Host'], 'Sequence': row['HA_seq']}, axis=1)
    df = df.drop(['HA_seq', 'Host'], axis=1)
    return df

df = create_classification_column(df)

# Convert to Huggingface Dataset
hf = Dataset.from_pandas(df)
hf = hf.train_test_split(train_size=0.8, seed=42)
hf_clean = hf["train"].train_test_split(train_size=0.8, seed=42)
hf_clean["validation"] = hf_clean.pop("test")
hf_clean["test"] = hf["test"]
hf = hf_clean

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/models/virusTrained100k')
source_lang = "Sequence"
target_lang = "Host"
prefix = "Sequence Host:"

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["classification"]]
    targets = [example[target_lang] for example in examples["classification"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=1737, truncation=False)
    return model_inputs

tokenized_hf = hf.map(preprocess_function, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model='/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/models/virusTrained100k')

# Initialize the model
model = T5ForConditionalGeneration.from_pretrained('/mmfs1/projects/changhui.yan/vishwajeet.marathe/InfluenzaProject/influenzaproject/models/virusTrained100k', from_flax=True)

# Evaluation metric
metric = evaluate.load("accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    epoch = trainer.state.epoch
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    combined_labels = decoded_preds + decoded_labels
    le = preprocessing.LabelEncoder()
    le.fit(combined_labels)
    
    numeric_preds = le.transform(decoded_preds)
    numeric_labels = le.transform(decoded_labels)
    
    cm = confusion_matrix(numeric_labels, numeric_preds)
    plt.figure(figsize=(36, 24))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Host")
    plt.savefig(f"confusion_matrix_{epoch}.png")
    
    accuracy = metric.compute(predictions=numeric_preds, references=numeric_labels)
    return accuracy

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="host_classification",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=13,
    predict_with_generate=True,
    fp16=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_hf['train'],
    eval_dataset=tokenized_hf['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save training history
train_metrics = trainer.state.log_history
train_df = pd.DataFrame(train_metrics)
train_df.to_csv("host_13_epochs_metrics_influenza.csv", index=False)
