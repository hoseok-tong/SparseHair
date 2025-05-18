import torch
import torch.nn as nn

from torch.nn import functional as F


class ComputeLossStrandVAE(nn.Module):
    def __init__(self, loss_dict):
        super().__init__()
        self.loss_dict = loss_dict

    def update_loss_weights(self, new_loss_dict):
        self.loss_dict.update(new_loss_dict)

    def __call__(self, net, p):
        p_, mu, logvar, _ = net(p)
        
        if torch.isnan(p_).sum() != 0:
            print('[Warning] Nan in the model output')
        
        _, dirsS = self.pos2dir(p)                         # gt
        _, dirsS_ = self.pos2dir(p_)                       # gen

        loss, log_dict = 0.0, {}
        
        for key, value in self.loss_dict.items():
            if key == 'l_main_mse':
                l_main_mse = F.mse_loss(p, p_) 

                loss += (l_main_mse * value)
                log_dict.update({f'{key}': l_main_mse * value})

            elif key == 'l_main_cos':
                l_main_cos = (1. - F.cosine_similarity(dirsS, dirsS_)).mean()
                
                loss += (l_main_cos * value)
                log_dict.update({f'{key}': l_main_cos*value})

            elif key == 'l_kld':
                if mu is None:
                    pass
                else:
                    l_kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
                    
                    loss += (l_kld * value)
                    log_dict.update({f'{key}': -l_kld*value})
            

        log_dict.update({f'loss': loss})
        return loss, log_dict

    def pos2dir(self, p):
        return 0, p[:, 1:, :] - p[:, :-1, :]
    
