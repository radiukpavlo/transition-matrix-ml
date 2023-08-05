import numpy as np
import torch


def generate_sample_dataset(input_dataset):
    """
    This function creates a new dataset subset with the
    specified number of samples.
    :param input_dataset: Original training dataset
    :return: Subset of the training dataset
            and a list of unique IDs (indices from the original dataset)
    """

    # Parameters
    n_samples = 1000

    generate_sample_labels = input_dataset.targets.numpy()

    # Determine the proportion of each class in the training dataset
    _, counts = np.unique(generate_sample_labels, return_counts=True)
    proportions = counts / len(generate_sample_labels)

    # Determine the number of samples to extract for each class
    samples_per_class = (proportions * n_samples).astype(int)

    # Adjust samples for any rounding issues to ensure exactly 1000 samples
    while np.sum(samples_per_class) < n_samples:
        class_with_max_samples = np.argmax(proportions)
        samples_per_class[class_with_max_samples] += 1

    # Extract samples based on the proportions
    indices_to_extract = []

    for gs_label, n_samples in enumerate(samples_per_class):
        label_indices = np.where(generate_sample_labels == gs_label)[0]
        chosen_indices = np.random.choice(label_indices, n_samples, replace=False)
        indices_to_extract.extend(chosen_indices)

    # Shuffle the indices for randomness
    np.random.shuffle(indices_to_extract)

    # Return a subset of the dataset
    sample_dataset = torch.utils.data.Subset(input_dataset, indices_to_extract)

    return sample_dataset, indices_to_extract
