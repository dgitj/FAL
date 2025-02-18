import torch
import torch.nn.functional as F

def sample(model, model_server, unlabeled_loader, c):
    model.eval()
    model_server.eval()
    discrepancy = torch.tensor([]).to(device)

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.to(device)

            scores, features = model(inputs)
            scores_t, _= model_server(inputs)

            # intensify specialized knowledge
            spc = loss_weight_list[c]
            spc = spc.unsqueeze(0).expand(labels.size(0), -1)

            const = 1
            scores = scores + const * spc.log()
            scores_t = scores_t + const * spc.log()
            scores = F.kl_div(F.log_softmax(scores, -1), F.softmax(scores_t, -1), reduction='none').sum(1) + F.kl_div(F.log_softmax(scores_t, -1), F.softmax(scores, -1), reduction='none').sum(1)
            scores *= 0.5
            discrepancy = torch.cat((discrepancy, scores), 0)

    return discrepancy.cpu()