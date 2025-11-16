"""
IFAL (Inconsistency-based Federated Active Learning) Strategy
Based on: "Inconsistency-Based Federated Active Learning" (IJCAI'25)

Complete implementation with Knowledge Distillation training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy

try:
    from skimage.filters import threshold_otsu
except ImportError:
    print("Warning: scikit-image not installed. Install with: pip install scikit-image")
    threshold_otsu = None

import config


class DatasetSplit(Dataset):
    """Dataset wrapper for subset of indices."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if config.DATASET == "PathMNIST":
            image = torch.from_numpy(self.dataset.imgs[self.idxs[item]]).permute(2, 0, 1).float()
            image = (image / 255.0 - 0.5) / 0.5
            label = self.dataset.labels[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label


def kd_loss(logits_student, logits_teacher, temperature):
    """Knowledge Distillation loss."""
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


class EarlyStopping:
    """Early stopping to prevent overfitting during KD training."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_wts = deepcopy(model.state_dict())
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_wts = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[IFAL KD] Validation loss has not improved for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_wts)


class IFALSampler:
    def __init__(self, device="cuda", top_k=250):
        """
        Initialize IFAL sampler.
        
        Args:
            device (str): Device to run calculations on
            top_k (int): Number of nearest neighbors for R-kNN
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("[IFAL] CUDA not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        
        self.top_k = top_k
        self.loss_func = nn.CrossEntropyLoss()
        
    def training_local_only(self, model_server, labeled_set, dataset):
        """
        Train a local model from scratch using only labeled data.
        
        Args:
            model_server: Global server model (to get architecture)
            labeled_set: Indices of labeled samples
            dataset: Training dataset
            
        Returns:
            torch.nn.Module: Trained local model
        """
        # Create a fresh local model
        local_net = deepcopy(model_server)
        
        # Reinitialize weights (train from scratch)
        for m in local_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        local_net = local_net.to(self.device)
        local_net.train()
        
        # Create dataloader
        label_train = DataLoader(DatasetSplit(dataset, labeled_set), 
                                batch_size=config.BATCH, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = optim.SGD(local_net.parameters(), 
                             lr=config.LR, 
                             momentum=config.MOMENTUM,
                             weight_decay=config.WDECAY)
        
        finetune_ep = 50
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[int(finetune_ep * 3 / 4)], 
                                                    gamma=0.1)
        
        # Training loop
        for epoch in range(finetune_ep):
            local_net.train()
            for images, labels in label_train:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                output = local_net(images)
                
                if isinstance(output, tuple):
                    output = output[0]
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)
                
                loss = self.loss_func(output, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Check for early convergence
            local_net.eval()
            correct, cnt = 0., 0.
            with torch.no_grad():
                for images, labels in label_train:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = local_net(images)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    y_pred = output.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                    cnt += len(labels)
            
            acc = correct / cnt
            if acc >= 0.99:
                print(f"[IFAL] Local-only training converged at epoch {epoch+1} with {acc:.3f} accuracy")
                break
        
        return local_net
    
    def training_local_only_KD(self, model_local, model_server, labeled_set, dataset):
        """
        Train a local model using Knowledge Distillation from global model.
        
        Args:
            model_local: Base local model (can be copy of global)
            model_server: Global server model (teacher)
            labeled_set: Indices of labeled samples
            dataset: Training dataset
            
        Returns:
            torch.nn.Module: KD-trained local model
        """
        global_net = deepcopy(model_server).to(self.device)
        local_net = deepcopy(model_local).to(self.device)
        
        global_net.eval()
        local_net.train()
        
        # Create dataloader
        label_train = DataLoader(DatasetSplit(dataset, labeled_set), 
                                batch_size=config.BATCH, shuffle=True)
        
        # Optimizer with smaller learning rate for KD
        optimizer = optim.SGD(local_net.parameters(), 
                             lr=config.LR * 0.1, 
                             momentum=config.MOMENTUM,
                             weight_decay=config.WDECAY)
        
        early_stopping = EarlyStopping(patience=5)
        
        finetune_ep = 500
        
        # Training loop with KD
        for epoch in range(finetune_ep):
            local_net.train()
            running_loss = 0.0
            
            for images, labels in label_train:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Student predictions
                output = local_net(images)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Teacher predictions
                with torch.no_grad():
                    global_output = global_net(images)
                    if isinstance(global_output, tuple):
                        global_output = global_output[0]
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)
                
                # Combined loss: CE + KD
                loss_ce = self.loss_func(output, labels)
                loss_kd_val = kd_loss(output, global_output, temperature=4.0)
                loss = 0.1 * loss_ce + 0.9 * loss_kd_val
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            running_loss /= len(label_train)
            early_stopping(running_loss, local_net)
            
            if early_stopping.early_stop:
                print(f"[IFAL] KD training early stopped at epoch {epoch+1}")
                early_stopping.load_best_model(local_net)
                break
        
        return local_net
        
    def get_rknn_label(self, labeled_set, unlabeled_set, model, dataset):
        """
        Get labels for labeled and unlabeled samples using model predictions.
        
        Args:
            labeled_set: Indices of labeled samples
            unlabeled_set: Indices of unlabeled samples  
            model: Model to use for predictions
            dataset: Dataset to get samples from
            
        Returns:
            list: All labels (ground truth for labeled, predicted for unlabeled)
        """
        all_labels = []
        
        # Get ground truth labels for labeled set
        for idx in labeled_set:
            if config.DATASET == "PathMNIST":
                label = dataset.labels[idx]
            else:
                _, label = dataset[idx]
            all_labels.append(int(label))
        
        # Get predicted labels for unlabeled set
        model.eval()
        with torch.no_grad():
            for idx in unlabeled_set:
                if config.DATASET == "PathMNIST":
                    image = torch.from_numpy(dataset.imgs[idx]).permute(2, 0, 1).float()
                    # Apply normalization
                    image = (image / 255.0 - 0.5) / 0.5
                else:
                    image, _ = dataset[idx]
                
                image = image.unsqueeze(0).to(self.device)
                output = model(image)
                if isinstance(output, tuple):
                    output = output[0]
                pred = output.max(1)[1]
                all_labels.append(pred.item())
        
        return all_labels
    
    def get_embedding_and_prob(self, data_idxs, model, dataset):
        """
        Get embeddings and probabilities for samples.
        
        Args:
            data_idxs: Indices of samples
            model: Model to use
            dataset: Dataset
            
        Returns:
            tuple: (embeddings, probabilities)
        """
        model.eval()
        embeddings = []
        probs = []
        
        with torch.no_grad():
            for idx in data_idxs:
                if config.DATASET == "PathMNIST":
                    image = torch.from_numpy(dataset.imgs[idx]).permute(2, 0, 1).float()
                    image = (image / 255.0 - 0.5) / 0.5
                else:
                    image, _ = dataset[idx]
                
                image = image.unsqueeze(0).to(self.device)
                output = model(image)
                
                # Handle different output formats
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                    # Try to get embeddings from second element
                    if len(output) > 1 and torch.is_tensor(output[1]):
                        emb = output[1]
                    else:
                        emb = logits  # fallback to logits
                else:
                    logits = output
                    emb = output
                
                embeddings.append(emb.squeeze())
                probs.append(F.softmax(logits, dim=1).squeeze())
        
        embeddings = torch.stack(embeddings)
        probs = torch.stack(probs)
        
        return embeddings, probs
    
    def compute_rknn_logits(self, features_labeled, features_unlabeled, labelArr_all):
        """
        Compute reverse k-NN logits for unlabeled samples.
        
        Args:
            features_labeled: Features of labeled samples
            features_unlabeled: Features of unlabeled samples
            labelArr_all: Labels of all samples (labeled + unlabeled predictions)
            
        Returns:
            torch.Tensor: R-kNN based logits
        """
        # Normalize features
        features_labeled = F.normalize(features_labeled, dim=1)
        features_unlabeled = F.normalize(features_unlabeled, dim=1)
        
        # Compute similarity matrix
        all_features = torch.vstack((features_labeled, features_unlabeled))
        dists_all = torch.mm(all_features, features_unlabeled.t())
        
        # Mask self-similarity for unlabeled samples
        start_idx = features_labeled.size(0)
        for i in range(features_unlabeled.size(0)):
            dists_all[start_idx + i, i] = -1
        
        # Get top-k similar samples
        top_k = min(self.top_k, dists_all.size(1))
        _, top_k_index = dists_all.topk(top_k, dim=1, largest=True, sorted=True)
        
        # Build R-kNN logits
        rknn_logits = torch.ones(features_unlabeled.shape[0], config.NUM_CLASSES, 
                                 dtype=torch.long).to(self.device)
        
        labelArr_all = np.array(labelArr_all)
        for i in range(config.NUM_CLASSES):
            class_mask = labelArr_all == i
            if class_mask.sum() > 0:
                unique_indices, counts = torch.unique(
                    top_k_index[class_mask], return_counts=True
                )
                rknn_logits[unique_indices, i] += counts.to(self.device)
        
        return rknn_logits
    
    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, 
                      num_samples, labeled_set=None, seed=None, dataset=None):
        """
        Select samples using IFAL strategy with full 3-model approach.
        
        Args:
            model: Local client model
            model_server: Global server model
            unlabeled_loader: DataLoader for unlabeled data
            c: Client ID
            unlabeled_set: List of unlabeled sample indices
            num_samples: Number of samples to select
            labeled_set: List of labeled sample indices
            seed: Random seed
            dataset: Dataset object (needed for IFAL's R-kNN computation)
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        if threshold_otsu is None:
            print("[IFAL] Warning: Otsu thresholding unavailable, falling back to top-k selection")
            return self._fallback_selection(model, unlabeled_loader, unlabeled_set, num_samples)
        
        # Get dataset from loader if not provided
        if dataset is None:
            dataset = unlabeled_loader.dataset
        
        if labeled_set is None or len(labeled_set) == 0:
            print("[IFAL] Warning: No labeled set provided, using random selection")
            return self._fallback_selection(model, unlabeled_loader, unlabeled_set, num_samples)
        
        unlabeled_set = np.array(unlabeled_set)
        
        print(f"[IFAL] Client {c}: Training local models...")
        
        # === Train local-only model (from scratch) ===
        l_net = self.training_local_only(model_server, labeled_set, dataset)
        
        # === Train local-KD model ===
        l_net_kd = self.training_local_only_KD(model_server, model_server, labeled_set, dataset)
        
        print(f"[IFAL] Client {c}: Computing inconsistency scores for {len(unlabeled_set)} samples")
        
        # === MODEL 1: Global model predictions ===
        g_labelArr_all = self.get_rknn_label(labeled_set, unlabeled_set, model_server, dataset)
        g_features_l, _ = self.get_embedding_and_prob(labeled_set, model_server, dataset)
        g_features_un, g_c_probs = self.get_embedding_and_prob(unlabeled_set, model_server, dataset)
        
        # Compute R-kNN logits for global model
        g_rknn_logits = self.compute_rknn_logits(g_features_l, g_features_un, g_labelArr_all)
        g_probs1 = (g_rknn_logits / g_rknn_logits.sum(1, keepdim=True)).cpu()
        g_probs2 = g_probs1  # In original code, same as g_probs1
        g_c_pred = g_c_probs.max(1)[1].cpu()
        g_pred1 = g_probs1.max(1)[1]
        g_pred2 = g_probs2.max(1)[1]
        
        # === MODEL 2: Local model predictions ===
        l_labelArr_all = self.get_rknn_label(labeled_set, unlabeled_set, l_net, dataset)
        l_features_l, _ = self.get_embedding_and_prob(labeled_set, l_net, dataset)
        l_features_un, l_c_probs = self.get_embedding_and_prob(unlabeled_set, l_net, dataset)
        
        # Compute R-kNN logits using global labels (for consistency)
        l_rknn_logits_g = self.compute_rknn_logits(l_features_l, l_features_un, g_labelArr_all)
        l_probs1 = (l_rknn_logits_g / l_rknn_logits_g.sum(1, keepdim=True)).cpu()
        
        # Compute R-kNN logits using local labels
        l_rknn_logits_l = self.compute_rknn_logits(l_features_l, l_features_un, l_labelArr_all)
        l_probs2 = (l_rknn_logits_l / l_rknn_logits_l.sum(1, keepdim=True)).cpu()
        
        l_c_pred = l_c_probs.max(1)[1].cpu()
        l_pred1 = l_probs1.max(1)[1]
        l_pred2 = l_probs2.max(1)[1]
        
        # === MODEL 3: Local-KD model predictions ===
        l_kd_labelArr_all = self.get_rknn_label(labeled_set, unlabeled_set, l_net_kd, dataset)
        l_kd_features_l, _ = self.get_embedding_and_prob(labeled_set, l_net_kd, dataset)
        l_kd_features_un, l_kd_c_probs = self.get_embedding_and_prob(unlabeled_set, l_net_kd, dataset)
        
        # Compute R-kNN logits using global labels
        l_kd_rknn_logits_g = self.compute_rknn_logits(l_kd_features_l, l_kd_features_un, g_labelArr_all)
        l_kd_probs1 = (l_kd_rknn_logits_g / l_kd_rknn_logits_g.sum(1, keepdim=True)).cpu()
        
        # Compute R-kNN logits using local-kd labels
        l_kd_rknn_logits_l = self.compute_rknn_logits(l_kd_features_l, l_kd_features_un, l_kd_labelArr_all)
        l_kd_probs2 = (l_kd_rknn_logits_l / l_kd_rknn_logits_l.sum(1, keepdim=True)).cpu()
        
        l_kd_c_pred = l_kd_c_probs.max(1)[1].cpu()
        l_kd_pred1 = l_kd_probs1.max(1)[1]
        l_kd_pred2 = l_kd_probs2.max(1)[1]
        
        # === Compute inconsistency scores (following original implementation) ===
        # D_ll: Wasserstein distance between local (with global labels) and local-KD (with global labels)
        D_ll = torch.tensor([wasserstein_distance(l_probs1[i], l_kd_probs1[i]) 
                            for i in range(l_probs1.shape[0])])
        
        # D_gl: Wasserstein distance between global and local-KD (both with global labels)
        D_gl = torch.tensor([wasserstein_distance(g_probs1[i], l_kd_probs1[i]) 
                            for i in range(l_probs1.shape[0])])
        
        print(f"[IFAL] D_ll range: [{D_ll.min():.4f}, {D_ll.max():.4f}], mean: {D_ll.mean():.4f}")
        print(f"[IFAL] D_gl range: [{D_gl.min():.4f}, {D_gl.max():.4f}], mean: {D_gl.mean():.4f}")
        print(f"[IFAL] Ratio range: [{torch.max(D_gl/(D_ll+1e-5), D_ll/(D_gl+1e-5)).min():.4f}, "
              f"{torch.max(D_gl/(D_ll+1e-5), D_ll/(D_gl+1e-5)).max():.4f}]")
        
        # U1: Combined inconsistency between global and local
        U1 = torch.max(D_gl / (D_ll + 1e-5), D_ll / (D_gl + 1e-5)) * (D_ll + D_gl)
        U1 = U1.numpy()
        
        # U2: Wasserstein distance between local R-kNN (with local labels) and local classifier
        U2 = torch.tensor([wasserstein_distance(l_probs2[i], l_c_probs.cpu()[i]) 
                          for i in range(l_probs2.shape[0])])
        U2 = U2.numpy()
        
        # Combined uncertainty
        U = U1 * U2
        
        # Print prediction accuracies
        print(f"[IFAL] Global model prediction accuracy: {(g_pred2 == g_c_pred).sum().item() / g_c_pred.shape[0]:.3f}")
        print(f"[IFAL] Local model prediction accuracy: {(l_pred2 == l_c_pred).sum().item() / l_c_pred.shape[0]:.3f}")
        print(f"[IFAL] Local KD model prediction accuracy: {(l_kd_pred2 == l_kd_c_pred).sum().item() / l_kd_c_pred.shape[0]:.3f}")
        
        # === Select samples using two-stage Otsu thresholding ===
        try:
            th = threshold_otsu(U.reshape(-1, 1))
            print(f"[IFAL] First stage Otsu threshold (U1*U2): {th:.4f}")
            
            candidate_mask = U > th
            candidate_features = l_features_un.cpu()[candidate_mask]
            candidate_idxs = unlabeled_set[candidate_mask]
            
            if len(candidate_idxs) > 0:
                print(f"[IFAL] Selected {len(candidate_idxs)} candidates above threshold")
                selected_idxs = self._kmeans_sample(
                    min(num_samples, len(candidate_idxs)), candidate_features
                )
                selected_idxs = candidate_idxs[selected_idxs]
            else:
                selected_idxs = np.array([], dtype=np.int64)
            
            # Second stage: If not enough samples, fall back to U1 only
            if len(selected_idxs) < num_samples:
                print(f"[IFAL] Only found {len(selected_idxs)} samples, using second stage (U1 only)")
                U = U1  # Use U1 only
                th = threshold_otsu(U.reshape(-1, 1))
                print(f"[IFAL] Second stage Otsu threshold (U1): {th:.4f}")
                
                candidate_mask = U > th
                candidate_features = l_features_un.cpu()[candidate_mask]
                candidate_idxs = unlabeled_set[candidate_mask]
                
                if len(candidate_idxs) > 0:
                    print(f"[IFAL] Second stage: {len(candidate_idxs)} candidates")
                    selected_idxs = self._kmeans_sample(
                        min(num_samples, len(candidate_idxs)), candidate_features
                    )
                    selected_idxs = candidate_idxs[selected_idxs]
                else:
                    # Last resort: select by highest combined uncertainty
                    print(f"[IFAL] Last resort: selecting by highest U1*U2")
                    selected_idxs = unlabeled_set[np.argsort(U1 * U2)[-num_samples:]]
        
        except Exception as e:
            print(f"[IFAL] Error in Otsu thresholding: {e}, using fallback")
            # Fallback: select by highest uncertainty
            selected_idxs = unlabeled_set[np.argsort(U1 * U2)[-num_samples:]]
        
        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_idxs]
        
        print(f"[IFAL] Client {c}: Selected {len(selected_idxs)} samples")
        
        return list(selected_idxs), remaining_unlabeled
    
    def _kmeans_sample(self, n, feats):
        """
        Sample n representative samples using KMeans clustering.
        
        Args:
            n: Number of samples to select
            feats: Feature vectors
            
        Returns:
            np.array: Indices of selected samples
        """
        feats = feats.numpy() if isinstance(feats, torch.Tensor) else feats
        
        cluster_learner = KMeans(n_clusters=n, random_state=0)
        cluster_learner.fit(feats)
        
        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = ((feats - centers) ** 2).sum(axis=1)
        
        # Select sample closest to each cluster center
        return np.array([
            np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] 
            for i in range(n) if (cluster_idxs == i).sum() > 0
        ])
    
    def _fallback_selection(self, model, unlabeled_loader, unlabeled_set, num_samples):
        """
        Fallback selection using entropy-based uncertainty.
        
        Args:
            model: Model to use
            unlabeled_loader: DataLoader
            unlabeled_set: Unlabeled indices
            num_samples: Number to select
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        model.eval()
        entropies = []
        
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = F.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                entropies.extend(entropy.cpu().numpy())
        
        entropies = np.array(entropies)
        selected_indices = np.argsort(entropies)[-num_samples:]
        selected_samples = [unlabeled_set[i] for i in selected_indices]
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        return selected_samples, remaining_unlabeled