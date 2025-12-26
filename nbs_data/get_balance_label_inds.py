import sys
sys.path.append('.')

from dataset import InstrDataset
import numpy as np

train_val_test_ratio = [0.7, 0.1, 0.2]

def get_balance_label_inds(all_labels, train_val_test_ratio):
    all_labels = np.array(all_labels)
    unique_labels = np.unique(all_labels)
    label_to_inds = {label: np.where(all_labels == label)[0] for label in unique_labels}
    
    train_inds, val_inds, test_inds = [], [], []
    
    for label, inds in label_to_inds.items():
        np.random.shuffle(inds)
        n_total = len(inds)
        n_train = int(n_total * train_val_test_ratio[0])
        n_val = int(n_total * train_val_test_ratio[1])
        
        train_inds.extend(inds[:n_train])
        val_inds.extend(inds[n_train:n_train + n_val])
        test_inds.extend(inds[n_train + n_val:])
    
    np.random.shuffle(train_inds)
    np.random.shuffle(val_inds)
    np.random.shuffle(test_inds)
    
    return train_inds, val_inds, test_inds

if __name__ == "__main__":
    all_labels = []
    dataset_name = 'EHBS'
    target_name = 'Asym-AD'
    dataset = InstrDataset(dataset_name, target_name=target_name, patch_size=32, next_time_mask=True, use_random_prompt=True)
    for i in range(len(dataset)):
        label = dataset[i][-1]
        all_labels.append(label)

    train_inds, val_inds, test_inds = get_balance_label_inds(all_labels, train_val_test_ratio)
    np.save(f'data/{dataset_name}/fmri/label_inds_{target_name}.npy', {'train_inds': train_inds, 'val_inds': val_inds, 'test_inds': test_inds}) 