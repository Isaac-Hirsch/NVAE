import torch
import torchvision
from ot.sliced import sliced_wasserstein_distance
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

def _sub_recon_flat_gen(concepts: str, data_loader, model, num_batches=20):
    # TODO test wasserstein distance between dataload and itself, should be near 0 even if has different batches

    model.eval()
    batch_label = concepts.split('-')
    orig_batches = []
    recon_batches = []
    i = 0
    print(f'concept: {concepts}')
    while i < num_batches:
        for (batch, concepts_label) in data_loader:
            if concepts_label != concepts:
                continue
            i += 1
            print(f"Processing batch {i+1}/{num_batches} for concepts {concepts}")
            if i >= num_batches:
                break
            bsz = batch.size(0)
            flat = batch.view(bsz, -1)
            orig_batches.append(flat)
            with torch.no_grad():
                logits = model.module.sample(num_samples=bsz, t=1.0, batch_label=batch_label)
                output = model.module.decoder_output(logits)
                if isinstance(output, torch.distributions.bernoulli.Bernoulli):
                    generated = output.mean
                else:
                    generated = output.sample()
            recon_batches.append(generated.view(bsz, -1))

        orig_flat = torch.cat(orig_batches, dim=0)
        recon_flat = torch.cat(recon_batches, dim=0).cpu()
    return orig_flat, recon_flat

def _compute_ot_gen(concepts: str, data_loader, model, num_batches=20):
    orig, recon = _sub_recon_flat_gen(concepts, data_loader, model, num_batches)
    return sliced_wasserstein_distance(orig, recon).item()

def compute_ood_metrics(concepts, data_loader, model, path, ood: bool = True):
    # Initialize a list to store metrics
    metrics_list = []

    for concept in concepts:
        # Create a DataLoader for the current concept
        split_concept = concept
        # Compute metrics for the current concept
        ot_gen_value = _compute_ot_gen(split_concept, data_loader, model, num_batches=100)

        # Append the metrics as a dictionary to the list
        metric_name = "ood" if ood else "id"
        metrics_list.append(
            {
                "concept": concept,
                f"{metric_name}_ot_gen": ot_gen_value
            }
        )

    # Convert the list of dictionaries to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(f"{path}/{metric_name}_metrics.csv", index=False)

class DictDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.lengths = [len(data) for data in data_dict.values()]
        self.total_length = sum(self.lengths)

    def __len__(self):
        if arch_flag == "vanilla-obs":
            return len(self.data_dict["obs"])
        return self.total_length

    def __getitem__(self, item):
        idx, key = item
        dataset = self.data_dict[key]
        return dataset[idx], key


class DictBatchSampler:
    def __init__(self, data_dict, batch_size):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.batch_size = batch_size
        self.total_samples = sum(len(dataset) for dataset in data_dict.values())
        if arch_flag == "vanilla-obs":
            self.total_samples = len(self.data_dict["obs"])

    def __iter__(self):
        samples_yielded = 0
        while samples_yielded < self.total_samples:
            key = random.choice(self.keys)
            if arch_flag == "vanilla-obs":
                key = "obs"
            dataset = self.data_dict[key]
            remaining = min(self.batch_size, self.total_samples - samples_yielded)
            indices = torch.randperm(len(dataset))[:remaining]
            yield [(idx.item(), key) for idx in indices]
            samples_yielded += len(indices)

    def __len__(self):
        return (self.total_samples + self.batch_size - 1) // self.batch_size


def dict_collate_fn(batch):
    data = torch.stack(
        [item[0][0] if isinstance(item[0], tuple) else item[0] for item in batch]
    )
    key = batch[0][1]  # All keys in the batch are the same
    return data, key

class ood_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        concept_combos = os.listdir(data_path)
        self.data = []
        self.labels = []
        for combo in concept_combos:
            print(f"Loading data for concept combo: {combo}")
            combo_path = os.path.join(data_path, combo, 'images')
            if os.path.isdir(combo_path):
                self.data += [os.path.join(combo_path, file) for file in os.listdir(combo_path) if file.endswith('.png')]
                self.labels += [combo for file in os.listdir(combo_path) if file.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = torchvision.io.decode_image(image_path)[:3, :, :]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loader(data_path, transform):
    dataset = ood_dataset(data_path, transform)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True, collate_fn=dict_collate_fn, drop_last=True)
    return data_loader

"""
# h_datasets are the heldout data from each concept
h_dataset = DictDataset(h_datasets)
batch_sampler = DictBatchSampler(h_datasets, batch_size=batch_size)
h_loader = DataLoader(
        h_dataset, batch_sampler=batch_sampler, collate_fn=dict_collate_fn
    )

    
single = ['obs', 'quad2', 'orientation', 'quad1', 'quad3', 'quad4', 'size']
double = ['quad2_orientation', 'quad2_quad3', 'quad2_quad4', 'quad1_orientation', 'quad2_size', 'quad1_quad2', 'quad1_quad3', 'quad1_quad4', 'quad1_size']

compute_metrics(single, h_loader, model, save_dir)
compute_ood_metrics(double, h_loader, model, save_dir)
"""