from transformers import AutoTokenizer, RobertaForMaskedLM, ElectraForMaskedLM, BertForMaskedLM, AutoConfig, DataCollatorWithPadding, DataCollatorForLanguageModeling
import torch
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RobertaForMaskedLM.from_pretrained('klue/roberta-large')
model.to(device)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='train_val.txt',
    block_size=514
)

data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2 # 0.2올려보기
)

training_args = TrainingArguments(
    output_dir="./klue-roberta-retrained",
    overwrite_output_dir=True,
    learning_rate=5e-05,
    num_train_epochs=200, # 학습을 길게 해보고 , early stopping 
    per_device_train_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    seed=30,
    save_strategy='epoch',
    gradient_accumulation_steps=8,
    logging_steps=100,
    evaluation_strategy='epoch',
    resume_from_checkpoint=True,
    fp16=True,
    fp16_opt_level='O1',
    load_best_model_at_end=True
) 

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model("./klue-roberta-retrained")