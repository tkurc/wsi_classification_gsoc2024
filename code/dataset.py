import os
import pandas as pd
from datasets import Dataset, DatasetDict

class ThymDataset:
    def __init__(self, root_dir='./thym'):
        self.root_dir = root_dir
        self.sub_dirs = ['train', 'test', 'val']
        self.class_dirs = {'til-positive': 1, 'til-negative': 0}
        self.data = []
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        self.df = self.load_data()
        
    def load_data(self):
        for sub_dir in self.sub_dirs:
            for class_dir, label in self.class_dirs.items():
                class_path = os.path.join(self.root_dir, sub_dir, class_dir)
                
                for file_name in os.listdir(class_path):
                    if file_name.lower().endswith(tuple(self.image_extensions)):
                        file_path = os.path.join(class_path, file_name)
                        self.data.append([file_path, label])
        
        df = pd.DataFrame(self.data, columns=['file_path', 'label'])
        if not df.empty:
            print("Data loaded successfully")
            return df
        else:
            print("Data loading failed")
            return pd.DataFrame(columns=['file_path', 'label'])

    def convert_to_huggingface_dataset_format(self):
        if not self.df.empty:
            dataset = Dataset.from_pandas(self.df)
            
            train_test_valid = dataset.train_test_split(test_size=0.3, seed=42)
            test_valid = train_test_valid['test'].train_test_split(test_size=0.6, seed=42)

            dataset_dict = DatasetDict({
                'train': train_test_valid['train'],
                'validation': test_valid['test'],
                'test': test_valid['train']
            })

            return dataset_dict
        else:
            print("Conversion failed: DataFrame is empty")
            return None
















"""
import os
import pandas as pd
from datasets import Dataset, DatasetDict

class ThymDataset:
    def __init__(self, root_dir='./thym'):
        self.root_dir = root_dir
        self.sub_dirs = ['train', 'test', 'val']
        self.class_dirs = {'til-positive': 1, 'til-negative': 0}
        self.data = []
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        self.dataset_dict = self.load_and_convert_data()

    def load_and_convert_data(self):
        for sub_dir in self.sub_dirs:
            for class_dir, label in self.class_dirs.items():
                class_path = os.path.join(self.root_dir, sub_dir, class_dir)
                
                for file_name in os.listdir(class_path):
                    if file_name.lower().endswith(tuple(self.image_extensions)):
                        file_path = os.path.join(class_path, file_name)
                        self.data.append([file_path, label])
        
        df = pd.DataFrame(self.data, columns=['file_path', 'label'])
        if not df.empty:
            print("Data loaded successfully")
            dataset = Dataset.from_pandas(df)
            
            train_test_valid = dataset.train_test_split(test_size=0.3, seed=42)
            test_valid = train_test_valid['test'].train_test_split(test_size=0.6, seed=42)

            dataset_dict = DatasetDict({
                'train': train_test_valid['train'],
                'validation': test_valid['test'],
                'test': test_valid['train']
            })

            return dataset_dict
        else:
            print("Data loading failed or DataFrame is empty")
            return None

# Usage example
thym_dataset = ThymDataset()
dataset_dict = thym_dataset.dataset_dict

"""