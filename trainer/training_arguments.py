from transformers import TrainingArguments


def init_tarining_arguments(evaluation_strategy, training_arguments_config, hyperparameter_config):
    if evaluation_strategy == 'epoch':
        training_args = init_epoch_training_arguments(
            training_arguments_config, hyperparameter_config)
    elif evaluation_strategy == 'steps':
        training_args = init_steps_training_arguments(
            training_arguments_config, hyperparameter_config)
    return training_args


def init_epoch_training_arguments(training_arguments_config, hyperparameter_config):
    training_args = TrainingArguments(
        output_dir=training_arguments_config['output_dir'],
        per_device_train_batch_size=hyperparameter_config['batch_size'],
        per_device_eval_batch_size=hyperparameter_config['batch_size'],
        gradient_accumulation_steps=hyperparameter_config['gradient_accumulation_steps'],
        learning_rate=hyperparameter_config['learning_rate'],
        weight_decay=hyperparameter_config['weight_decay'],
        num_train_epochs=hyperparameter_config['epochs'],
        logging_dir=training_arguments_config['logging_dir'],
        logging_steps=training_arguments_config['logging_step'],
        save_total_limit=training_arguments_config['save_total_limit'],
        evaluation_strategy=training_arguments_config['evaluation_strategy'],
        save_strategy=training_arguments_config['evaluation_strategy'],
        load_best_model_at_end=training_arguments_config['load_best_model_at_end'],
        metric_for_best_model=training_arguments_config['metric_for_best_model'],
        fp16=training_arguments_config['fp16'],
        fp16_opt_level=training_arguments_config['fp16_opt_level']
    )
    return training_args


def init_steps_training_arguments(training_arguments_config, hyperparameter_config):
    training_args = TrainingArguments(
        output_dir=training_arguments_config['output_dir'],
        per_device_train_batch_size=hyperparameter_config['batch_size'],
        per_device_eval_batch_size=hyperparameter_config['batch_size'],
        gradient_accumulation_steps=hyperparameter_config['gradient_accumulation_steps'],
        learning_rate=hyperparameter_config['learning_rate'],
        weight_decay=hyperparameter_config['weight_decay'],
        num_train_epochs=hyperparameter_config['epochs'],
        logging_dir=training_arguments_config['logging_dir'],
        logging_steps=training_arguments_config['logging_step'],
        save_total_limit=training_arguments_config['save_total_limit'],
        evaluation_strategy=training_arguments_config['evaluation_strategy'],
        eval_steps=training_arguments_config['eval_steps'],
        save_steps=training_arguments_config['save_steps'],
        load_best_model_at_end=training_arguments_config['load_best_model_at_end'],
        metric_for_best_model=training_arguments_config['metric_for_best_model'],
        fp16=training_arguments_config['fp16'],
        fp16_opt_level=training_arguments_config['fp16_opt_level']
    )
    return training_args
