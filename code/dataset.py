import os
import pandas as pd
from datasets import Dataset, DatasetDict

class TILDataset:
    """
    Attributes:
    -----------
    root_dir: str
        The root directory of the dataset
    sub_dirs: list
        sub directories of the dataset
    class_dirs: dict
        class directories of the dataset
    data: list
        list of data
    image_extensions: set
        set of image extensions
    df: pd.DataFrame
        pandas DataFrame of the dataset
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sub_dirs = ['train', 'test', 'val']
        self.class_dirs = {'til-positive': 1, 'til-negative': 0}
        self.data = []
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        self.df = self.load_data()
        
    def load_data(self):
        """
        Load data from the root directory and its subdirectories.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing file paths and labels.
        """
        for sub_dir in self.sub_dirs:
            sub_dir_path = os.path.join(self.root_dir, sub_dir)
            if os.path.exists(sub_dir_path):
                for class_dir, label in self.class_dirs.items():
                    class_path = os.path.join(sub_dir_path, class_dir)
                    if os.path.exists(class_path):
                        for file_name in os.listdir(class_path):
                            if file_name.lower().endswith(tuple(self.image_extensions)):
                                file_path = os.path.join(class_path, file_name)
                                self.data.append([file_path, label])
            else:
                for root_sub_dir in os.listdir(self.root_dir):
                    sub_dir_path = os.path.join(self.root_dir, root_sub_dir, sub_dir)
                    if os.path.exists(sub_dir_path):
                        for class_dir, label in self.class_dirs.items():
                            class_path = os.path.join(sub_dir_path, class_dir)
                            if os.path.exists(class_path):
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


# Load the Data
def load_dataset():

    """
    Load the Thym dataset and convert it to Hugging Face dataset format

    Parameters:
    -----------
    root_dir: str
        The root directory of the dataset

    Returns:
    --------
    dataset: Hugging Face dataset format

    """
    til_data = TILDataset(root_dir='datasets/brca')
    
    huggingface_dataset = til_data.convert_to_huggingface_dataset_format()
    return huggingface_dataset
