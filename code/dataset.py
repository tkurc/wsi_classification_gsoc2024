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
        

    ### TRAINING DATA ####
    def load_train_data(self):
        """
        Load training data from the root directory and its subdirectories.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing file paths and labels.
        """
        for root_sub_dir in os.listdir(self.root_dir):
            root_sub_dir_path = os.path.join(self.root_dir, root_sub_dir)
            if os.path.isdir(root_sub_dir_path):
                for root_sub_tvt_dir in os.listdir(root_sub_dir_path):
                    # Process only 'test' directory
                    if root_sub_tvt_dir.lower() == "train":
                        root_sub_tvt_dir_path = os.path.join(root_sub_dir_path, root_sub_tvt_dir)                    
                        if os.path.isdir(root_sub_tvt_dir_path):
                            for class_dir, label in self.class_dirs.items():
                                class_path = os.path.join(root_sub_tvt_dir_path, class_dir)
                                if os.path.isdir(class_path):
                                    for file_name in os.listdir(class_path):
                                        file_path = os.path.join(class_path, file_name)
                                        # Check if it's a file and has the correct extension
                                        if os.path.isfile(file_path) and file_name.lower().endswith(tuple(self.image_extensions)):
                                            self.data.append([file_path, label])
        
        df_train = pd.DataFrame(self.data, columns=['file_path', 'label'])
        if not df_train.empty:
            print("Data loaded successfully")
            return df_train
        else:
            print("Data loading failed")
            return pd.DataFrame(columns=['file_path', 'label'])

    ### VALDATION DATA ####
    def load_val_data(self):
        """
        Load training data from the root directory and its subdirectories.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing file paths and labels.
        """
        for root_sub_dir in os.listdir(self.root_dir):
            root_sub_dir_path = os.path.join(self.root_dir, root_sub_dir)
            if os.path.isdir(root_sub_dir_path):
                for root_sub_tvt_dir in os.listdir(root_sub_dir_path):
                    # Process only 'test' directory
                    if root_sub_tvt_dir.lower() == "val":
                        root_sub_tvt_dir_path = os.path.join(root_sub_dir_path, root_sub_tvt_dir)                    
                        if os.path.isdir(root_sub_tvt_dir_path):
                            for class_dir, label in self.class_dirs.items():
                                class_path = os.path.join(root_sub_tvt_dir_path, class_dir)
                                if os.path.isdir(class_path):
                                    for file_name in os.listdir(class_path):
                                        file_path = os.path.join(class_path, file_name)
                                        # Check if it's a file and has the correct extension
                                        if os.path.isfile(file_path) and file_name.lower().endswith(tuple(self.image_extensions)):
                                            self.data.append([file_path, label])
        
        df_val = pd.DataFrame(self.data, columns=['file_path', 'label'])
        if not df_val.empty:
            print("Data loaded successfully")
            return df_val
        else:
            print("Data loading failed")
            return pd.DataFrame(columns=['file_path', 'label'])

    #### TEST DATA ####
    def load_test_data(self):
        """
        Load training data from the root directory and its subdirectories.

        Returns:
        --------
        pd.DataFrame
            DataFrame containing file paths and labels.
        """
        for root_sub_dir in os.listdir(self.root_dir):
            root_sub_dir_path = os.path.join(self.root_dir, root_sub_dir)
            if os.path.isdir(root_sub_dir_path):
                for root_sub_tvt_dir in os.listdir(root_sub_dir_path):
                    # Process only 'test' directory
                    if root_sub_tvt_dir.lower() == "test":
                        root_sub_tvt_dir_path = os.path.join(root_sub_dir_path, root_sub_tvt_dir)                    
                        if os.path.isdir(root_sub_tvt_dir_path):
                            for class_dir, label in self.class_dirs.items():
                                class_path = os.path.join(root_sub_tvt_dir_path, class_dir)
                                if os.path.isdir(class_path):
                                    for file_name in os.listdir(class_path):
                                        file_path = os.path.join(class_path, file_name)
                                        # Check if it's a file and has the correct extension
                                        if os.path.isfile(file_path) and file_name.lower().endswith(tuple(self.image_extensions)):
                                            self.data.append([file_path, label])
        
        df_test = pd.DataFrame(self.data, columns=['file_path', 'label'])
        if not df_test.empty:
            print("Data loaded successfully")
            return df_test
        else:
            print("Data loading failed")
            return pd.DataFrame(columns=['file_path', 'label'])
    

    @staticmethod
    def convert_to_huggingface_dataset_format():
        
        # Create an instance of the TILDataset
        dataset_instance = TILDataset(root_dir='datasets/')

        # # Load the training data
        train_df = Dataset.from_pandas(dataset_instance.load_train_data())
        # # Load the validation data
        val_df = Dataset.from_pandas(dataset_instance.load_val_data())
        # Load the test data
        test_df = Dataset.from_pandas(dataset_instance.load_test_data())

        dataset_dict = DatasetDict({
            "train": train_df,
            "validation": val_df,
            "test": test_df
        })
        return dataset_dict


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
    huggingface_dataset = TILDataset(root_dir='datasets/').convert_to_huggingface_dataset_format()
    return huggingface_dataset

# d = load_dataset()
# print(d)
# print(d["train"]["file_path"][0])
# print(d["train"]["label"][0])
