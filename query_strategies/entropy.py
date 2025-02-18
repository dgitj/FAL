import torch
import torch.nn.functional as F

def sample(model, model_server, unlabeled_loader, c, loss_weight_list):
    """ Entropy-based sampling: Selects samples with the highest uncertainty """
    model.eval()
    entropy = torch.tensor([]).cuda()

    with torch.no_grad():
        for (inputs, _) in unlabeled_loader:
            inputs = inputs.cuda()
            scores, _ = model(inputs)
            probabilities = F.softmax(scores, dim=-1)

            # Compute entropy: H(x) = -sum(p(x) * log(p(x)))
            entropy_batch = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=1)

            entropy = torch.cat((entropy, entropy_batch), 0)

    return entropy.cpu()
