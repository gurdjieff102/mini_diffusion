import torch
import torch.nn as nn
import numpy as np
import math
import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model.Unet import UNet
import torchvision
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import save_image, make_grid
from lightning.pytorch.callbacks import ModelCheckpoint


def betas_for_alpha_bar(T, alpha_bar, max_beta):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    betas = np.array(betas)
    return torch.from_numpy(betas)

def ddpm_schedule(beta_start, beta_end, T, scheduler_type = 'cosine'):
    assert beta_start < beta_end < 1.0
    
    if scheduler_type == 'linear':
        betas = torch.linspace(beta_start, beta_end, T)

    if scheduler_type == 'cosine':
        betas = betas_for_alpha_bar(T, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, beta_end)

    if scheduler_type not in ['linear', 'cosine']:
        raise NotImplementedError(f'Unknown Beta Schedule {scheduler_type}.')
    
    sqrt_beta_t = torch.sqrt(betas)
    alpha_t = 1 - betas
    log_alpha_t = torch.log(alpha_t)
    sqrt_alpha_t_inv = 1 / torch.sqrt(alpha_t)

    alphabar_t = torch.cumsum(log_alpha_t, dim = 0).exp()
    sqrt_abar_t = torch.sqrt(alphabar_t)
    sqrt_abar_t1 = torch.sqrt(1 - alphabar_t)
    alpha_t_div_sqrt_abar = (1 - alpha_t) / sqrt_abar_t1

    return {
        'sqrt_beta_t': sqrt_beta_t,
        'alpha_t': alpha_t,
        'sqrt_alpha_t_inv': sqrt_alpha_t_inv,
        'alphabar_t': alphabar_t,
        'sqrt_abar_t': sqrt_abar_t,
        'sqrt_abar_t1': sqrt_abar_t1,
        'alpha_t_div_sqrt_abar': alpha_t_div_sqrt_abar
    } 

class DDPM(L.LightningModule):
    def __init__(self, betas, T = 500, dropout_p = 0.1):
        super().__init__() 
        for k, v in ddpm_schedule(betas[0], betas[1], T).items():
            self.register_buffer(k, v)

        self.T = T
        self.dropout_p = dropout_p
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.criterion = nn.MSELoss()

        self.num_cls = 10
        self.unet = UNet(1, 128, self.num_cls)

    def forward(self, x, label):
        timestep = torch.randint(1, self.T, (x.shape[0], )).cuda()
        noise = torch.randn_like(x)

        x_t = ((self.sqrt_abar_t[timestep].unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], x.shape[3]) * x
                 + self.sqrt_abar_t1[timestep, None, None, None] * noise)).float()
        ctx_mask = torch.bernoulli(torch.zeros_like(label) + self.dropout_p).float()

        timestep = (timestep / self.T).float()
        pred = self.unet(x_t, label, timestep, ctx_mask)
        return pred, noise

    def training_step(self, batch, idx):
        # self.on_train_epoch_end()
        img, class_lbl = batch
        noise, predict = self(img, class_lbl)
        loss = self.criterion(predict, noise)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        num_sample = 4 * self.num_cls
        ws = [0.0, 0.5, 1.0]

        if self.current_epoch % 2 == 0:
            with torch.no_grad():
                # for w_i, w in enumerate(ws):
                x1, xis = self.sample(num_sample, (1, 28, 28), self.num_cls, 1.0)
                fig, ax = plt.subplots(nrows = num_sample // self.num_cls, ncols = self.num_cls, sharex = True, sharey = True, figsize = (10, 4))
                ani = FuncAnimation(fig, self.animate_plot, fargs = [xis, ax], interval = 200, blit = False, repeat = True, frames = xis.shape[0])
                ani.save(f'./epoch_{self.current_epoch}.gif', dpi = 100, writer = PillowWriter(fps = 5))

                grid = make_grid(x1, nrow=self.num_cls, normalize=True, pad_value=1)
                save_path = f'./epoch_{self.current_epoch}_grid.png'
                save_image(grid, save_path)
                print('GIF Saved!')

    def animate_plot(self, i, xis, ax):
        plots = []
        num_sample = 4 * self.num_cls
        for row in range(num_sample // self.num_cls):
          for col in range(self.num_cls):
            ax[row, col].clear()
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            plots.append(ax[row, col].imshow(-xis[i, (row*self.num_cls) + col, 0], cmap = 'gray', vmin = (-xis[i]).min(), vmax = (-xis[i]).max()))
        return plots
        
    def train_dataloader(self):
        dataset = MNIST(
            "./data",
            train=True,
            download=True,
            transform=self.transforms,
        )
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-4)

    def sample(self, num_samples, size, num_cls, guide_w = 0.0):
        x = torch.randn(num_samples, *size).cuda() 
        label = torch.arange(0, num_cls).cuda()
        label = label.repeat(int(num_samples / label.shape[0]))
        print(x.shape, label.shape)

        # ctx_mask = torch.zeros_like(label).cuda()
        label = label.repeat(2)
        # ctx_mask = ctx_mask.repeat(2)
        # ctx_mask[num_samples:] = 1.0

        #To Store intermediate results and create GIFs.
        x_is = []
        for i in range(self.T - 1, 0, -1):
            timestep = torch.tensor([i / self.T]).cuda()
            timestep = timestep.repeat(num_samples, 1, 1, 1)

            x = x.repeat(2, 1, 1, 1)   
            timestep = timestep.repeat(2, 1, 1, 1)
            z = torch.randn(num_samples, *size).cuda() if i > 1 else 0

            eps = self.unet(x, label, timestep)
            eps1 = eps[:num_samples]
            eps2 = eps[num_samples:]
            eps = (1 + guide_w)*eps1 - guide_w*eps2
            
            x = x[:num_samples]
            x = (self.sqrt_alpha_t_inv[i] * (x - eps*self.alpha_t_div_sqrt_abar[i]) + self.sqrt_beta_t[i] * z)

            if i % 25 == 0 or i == self.T - 1:
                x_is.append(x.detach().cpu().numpy())
        return x, np.array(x_is)

checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="train_loss",
        filename="language-{epoch:02d}-{train_loss:.4f}",
        save_top_k=3,
        mode="min",
    )    

trainer = L.Trainer(
    max_epochs=100,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
    accelerator='gpu' if torch.cuda.is_available else 'cpu',
    devices='auto'
)

ckpt_path = 'checkpoints/language-epoch=57-train_loss=0.0238.ckpt'
ddpm_model = DDPM((1e-4, 0.02))
trainer.fit(ddpm_model, ckpt_path=ckpt_path)


