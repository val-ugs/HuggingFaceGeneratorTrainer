from datasets import load_dataset
import numpy as np
import pandas as pd

from transformers import TrainingArguments
from GeneratorTrainer import GeneratorTrainer

def main():
    # For example, use eli5_category. And so you can use any Dataframe with a 'text' column
    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5 = eli5.flatten()

    texts = [" ".join(x) for x in eli5["answers.text"]]
    
    train, valid, test = np.split(texts, [int(len(texts)*0.8), int(len(texts)*0.9)]) # 80% train, 10% valid, 10% test
    
    train_df = pd.DataFrame(train, columns=['text'])
    valid_df = pd.DataFrame(valid, columns=['text'])
    test_df = pd.DataFrame(test, columns=['text'])

    output_dir = "trained_model" # path to unload the trained model

    #region training (comment on the region if you no longer need to train)
    
    # model (launch for the first time, next use fine-tuned model)
    generator_trainer = GeneratorTrainer(
        train_df=train_df,
        valid_df=valid_df,
        model_name_or_path="distilbert/distilgpt2"
    )
     
    # fine-tuned model (launch for additional training, don't forget comment above generator_trainer)
    # generator_trainer = GeneratorTrainer(
    #     train_df=train_df,
    #     valid_df=valid_df,
    #     model_name_or_path=output_dir
    # )

    training_args = TrainingArguments(
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=10,
        output_dir=output_dir,
    )

    generator_trainer.train(training_args, output_dir)
    #endregion

    #region generate (comment on the region if you don’t need to generate it)
    generator_trainer = GeneratorTrainer(
        train_df=train_df,
        valid_df=valid_df,
        model_name_or_path=output_dir,
    )

    # generate sentence
    prompt = "Somatic hypermutation allows the immune system to"
    print(generator_trainer.generate_sentence(prompt))

    # generate test_df
    test_df['text'] = test_df.apply(lambda row: " ".join(row['text'].split()[:7]), axis=1) # get 7 words in every row
    print(generator_trainer.generate(test_df))
    #endregion

if __name__ == '__main__': 
    main()