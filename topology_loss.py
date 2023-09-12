import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import topologylayer.nn
from topologylayer.nn import LevelSetLayer2D, BatchedLevelSetLayer2D,\
    SumBarcodeLengths, PartialSumBarcodeLengths


# generate circle on grid
# generate circle on grid
n = 36


def circlefn(i, j, n):
    r = np.sqrt((i - n/2.)**2 + (j - n/2.)**2)
    return np.exp(-(r - n/3.)**2/(n*2))


def gen_circle(n):
    beta = torch.empty(1, n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            beta[:, i, j] = circlefn(i, j, n)
    return beta

import ipdb; ipdb.set_trace()
beta = gen_circle(n)

m = 1024
b = 8
X = torch.randn(b, m, n ** 2, dtype=torch.float32)
y = X @ beta.view(-1, n ** 2, 1) + 0.05 * torch.randn(b, m, 1)
beta_ols = (torch.linalg.lstsq(X, y, rcond=None)[0]).view(b, n, n)


class TopLoss(nn.Module):
    def __init__(self, size):
        super(TopLoss, self).__init__()
        self.pdfn = BatchedLevelSetLayer2D(size=size, sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=1)
        self.topfn2 = SumBarcodeLengths(dim=0)

    def forward(self, beta):
        import ipdb; ipdb.set_trace()
        dgminfo = self.pdfn(beta)
        return (self.topfn(dgminfo, sumdim=1) + self.topfn2(dgminfo, sumdim=1)).mean()


tloss = TopLoss((n, n)) # topology penalty
dloss = nn.MSELoss() # data loss

beta_t = torch.autograd.Variable(torch.tensor(beta_ols).type(torch.float32), requires_grad=True)
X_t = torch.tensor(X, dtype=torch.float32, requires_grad=False)
y_t = torch.tensor(y, dtype=torch.float32, requires_grad=False)
optimizer = torch.optim.Adam([beta_t], lr=1e-2)
for _ in (pbar := tqdm(range(256))):
    optimizer.zero_grad()
    tlossi = tloss(beta_t)
    dlossi = dloss(y_t, X_t @ beta_t.view(b, -1, 1))
    loss = 0.1 * tlossi + dlossi
    loss.backward()
    optimizer.step()
    pbar.set_postfix_str(
        f"Topo loss {float(tlossi):.3f}, Recon loss {float(dlossi):.3f}")


# save figure
beta_est = beta_t.detach().numpy()
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for bind in range(3):
    ax[bind][0].imshow(beta[0, ...])
    ax[bind][0].set_title("Truth")
    ax[bind][1].imshow(beta_ols[bind, ...])
    ax[bind][1].set_title("OLS")
    ax[bind][2].imshow(beta_est[bind, ...])
    ax[bind][2].set_title("Topology Regu")
    for i in range(3):
        ax[bind][i].set_yticklabels([])
        ax[bind][i].set_xticklabels([])
        ax[bind][i].tick_params(bottom=False, left=False)

plt.savefig('noisy_circle.png')