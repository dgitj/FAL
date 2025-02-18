import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

def sample(model, model_server, unlabeled_loader, c, loss_weight_list, num_samples=100):
    """ BADGE: Selects samples based on uncertainty and diversity using gradient embeddings. """
    
    model.eval()
    device = next(model.parameters()).device

    embeddings = []
    data_indices = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
            inputs = inputs.to(device)

            # Get model predictions
            scores, _ = model(inputs)
            probabilities = F.softmax(scores, dim=-1)

            # Compute gradient embeddings: ∇L with respect to class probabilities
            max_probs, pseudo_labels = probabilities.max(dim=1)
            one_hot_labels = torch.zeros_like(probabilities).scatter_(1, pseudo_labels.unsqueeze(1), 1)
            grad_embeddings = (probabilities - one_hot_labels).detach().cpu().numpy()

            embeddings.append(grad_embeddings)
            data_indices.extend(range(batch_idx * len(inputs), (batch_idx + 1) * len(inputs)))

    # Stack all embeddings
    embeddings = np.vstack(embeddings)

    # Perform k-means++ clustering to select diverse points
    kmeans = KMeans(n_clusters=num_samples, init="k-means++", n_init=10, random_state=42)
    selected_indices = kmeans.fit(embeddings).labels_

    # Get indices of the selected samples
    selected_samples = np.array(data_indices)[selected_indices]

    return torch.tensor(selected_samples)
