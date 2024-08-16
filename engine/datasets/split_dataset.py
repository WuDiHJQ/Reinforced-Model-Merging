import os
import pickle
from torch.utils.data import Dataset

class Split_Dataset(Dataset):  
    def __init__(self, dataset, num_classes, classes_range, pkl_root):  
        
        self.dataset = dataset
        self.num_classes = num_classes
        self.pkl_root = pkl_root
        
        if os.path.exists(self.pkl_root):
            with open(self.pkl_root, 'rb') as f:
                self.indices = pickle.load(f)
        else:  
            self.indices = []
            for i, (_, label) in enumerate(self.dataset):
                if label in classes_range:
                    self.indices.append(i)
            with open(self.pkl_root, 'wb') as f:
                pickle.dump(self.indices, f)

    def __getitem__(self, idx):  
        sample_idx = self.indices[idx]  
        data, label = self.dataset[sample_idx]
        if label >= self.num_classes//2:
            label -= self.num_classes//2
        return data, label 
  
    def __len__(self):  
        return len(self.indices)