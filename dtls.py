import copy
import torch.nn.functional as F
import numpy as np
import glob
import shutil
import cv2
import os
import errno
import torch

from torch import nn
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam, AdamW
from torchvision import transforms, utils
from PIL import Image

from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

####### helpers functions

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class DTLS(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        stride,
        timesteps,
        device,
    ):
        super().__init__()
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.func = self.get_funcs()
        self.stride = stride
        self.device = device
    def transform_func(self, img, dec_size):
        img_1 = F.interpolate(img, size=img.shape[2] - dec_size * self.stride, mode='bilinear')
        img_1 = F.interpolate(img_1, size=img.shape[2], mode='bilinear')
        return img_1

    def get_funcs(self):
        all_funcs = []
        for i in range(self.num_timesteps + 1):
            all_funcs.append((lambda img, d=i: self.transform_func(img, d)))

        return all_funcs


    @torch.no_grad()
    def sample(self, batch_size = 16, img=None, t=None):
        if t==None:
            t=self.num_timesteps

        blur_img = self.func[t](img)

        img_t = blur_img.clone()
        previous_x_s0 = None
        momentum = 0

        ####### Domain Transfer
        while(t):
            print("Current step: ", t, "| shape: ", img.shape[2] - t * self.stride)

            step = torch.full((batch_size,), t, dtype=torch.long).to(self.device)

            if previous_x_s0 is None:
                momentum_l = 0
            else:
                momentum_l = self.func[t](momentum)

            R_x = self.denoise_fn(img_t + momentum_l, step)

            if previous_x_s0 is None:
                previous_x_s0 = R_x
            momentum += previous_x_s0 - R_x
            previous_x_s0 = R_x

            x4 = self.func[t-1](R_x.clone())
            img_t = x4

            t -=1
        return blur_img, img_t

    def p_losses(self, x_start, t):
        x_blur = x_start.clone()
        for i in range(t.shape[0]):
            x_blur[i] = self.func[t[i]](x_blur[i].unsqueeze(0))
        x_recon = self.denoise_fn(x_blur, t)

        ### Loss function
        loss = (x_start - x_recon).abs().mean()

        return loss

    def forward(self, x, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps + 1, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)

# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size*1.12), int(image_size*1.12))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        input_image,
        results_folder,
        load_path = None,
        shuffle=True,
        device,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.nrow = 4
        self.metrics_list = []
        self.input_image = input_image

        self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=1))
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)

        self.step = 0

        self.device = device
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)


    def train(self):

        backwards = partial(loss_backwards, self.fp16)
        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).to(self.device)
                loss = torch.mean(self.model(data))
                print(f'{self.step}: {loss.item()}')
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                batches = self.batch_size
                og_img = next(self.dl).to(self.device)
                blur_img, img_sr = self.ema_model.sample(batch_size=batches, img=og_img)

                utils.save_image(og_img, str(self.results_folder / f'{milestone}_original.png'), nrow=self.nrow)
                utils.save_image(img_sr, str(self.results_folder / f'{milestone}_SR.png'), nrow = self.nrow)
                utils.save_image(blur_img, str(self.results_folder / f'{milestone}_LR.png'), nrow = self.nrow)

                acc_loss = acc_loss/(self.save_and_sample_every+1)

                gt = cv2.imread(f'{self.results_folder}/{milestone}_original.png')
                sr = cv2.imread(f'{self.results_folder}/{milestone}_SR.png')

                psnr = compare_psnr(gt, sr, data_range=255)
                ssim = compare_ssim(gt, sr, channel_axis=-1)

                self.metrics_list.append(f"{self.step} PSNR/SSIM: {psnr} | {ssim}")

                file = open(f"{self.results_folder}/metrics.txt", 'w')
                for line in self.metrics_list:
                    file.write(line + "\n")
                file.close()

                print(f'Mean of last {self.step}: {acc_loss}')
                acc_loss=0

                self.save()

            self.step += 1

        print('training completed')

    def evaluation(self):
        for idx, path in enumerate(sorted(glob.glob(os.path.join(self.input_image, '*')))):
            imgname = os.path.splitext(os.path.basename(path))[0]
            print(idx, imgname)
            # read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
            img = img.unsqueeze(0).to(self.device)

            blur_img, img_sr = self.ema_model.sample(batch_size=1, img=img)

            utils.save_image(img_sr, str(self.results_folder / f'SR_{imgname}.png'), nrow=self.nrow)
