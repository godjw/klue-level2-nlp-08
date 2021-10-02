from transformers import AutoTokenizer, BertForMaskedLM, AutoConfig, DataCollatorWithPadding, DataCollatorForLanguageModeling
import torch
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForMaskedLM.from_pretrained('klue/bert-base')
model.to(device)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='train.txt',
    block_size=512
)

data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./kluebert-retrained",
    overwrite_output_dir=True,
    learning_rate=5e-05,
    num_train_epochs=15,
    per_device_train_batch_size=32,
    save_steps=100,
    save_total_limit=2,
    seed=42,
    save_strategy='steps',
    gradient_accumulation_steps=4,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./kluebert-retrained")