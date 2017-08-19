# vk_text_classifier
Neural network classifier of vk (or even facebook) profiles based on texts corpora.

Usage:
```
corpora_class = CorporaClass()

folders = ["corpora_train", "corpora_test"]
for folder_name in folders:
    for filename in tqdm.tqdm(os.listdir(folder_name)):
        with open(f"{folder_name}/{filename}") as f:
            corpora_class.add_to_corpora(f)
corpora_class.process_corpora()
```