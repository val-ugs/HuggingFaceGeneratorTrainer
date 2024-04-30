from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, TrainingArguments, Trainer
import math

class GeneratorTrainer:
    def __init__(
            self, 
            train_df, 
            valid_df,
            model_name_or_path):
        self.train_df = train_df
        self.valid_df = valid_df

        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenized_train_dataset = self.__get_dataset(train_df)
        self.tokenized_val_dataset = self.__get_dataset(valid_df)

    def __get_dataset(self, df):
        new_df = df.copy()
        dataset = Dataset.from_pandas(new_df)
        dataset = dataset.map(self.__preprocess_function, batched=True, remove_columns=new_df.columns.values)
        tokenized_dataset = dataset.map(self.__group_texts, batched=True)

        return tokenized_dataset

    def __preprocess_function(self, examples):
        return self.tokenizer(examples["text"])
    
    def __group_texts(self, examples, block_size = 128):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self, training_args, output_dir):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate(self, df):
        texts = []
        for _, row in df.iterrows():
            inputs = self.tokenizer(row['text'], truncation=True, padding='max_length', max_length=512,  return_tensors="pt").input_ids
            outputs = self.model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            texts.append(text)
        return texts

    def generate_sentence(self, prompt):
        inputs = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=512, return_tensors="pt").input_ids
        outputs = self.model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)