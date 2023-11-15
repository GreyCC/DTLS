import torch.cuda
from unet import Unet
from dtls import DTLS, Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--mode', default='train', type=str, help="mode for either 'train' or 'eval'")
parser.add_argument('--hr_size', default=128, type=int, help="size of HR image")
parser.add_argument('--lr_size', default=16, type=int, help="size of LR image")
parser.add_argument('--stride', default=4, type=int, help="size change between each step")
parser.add_argument('--train_steps', default=50000, type=int)
parser.add_argument('--lr_rate', default=2e-5, help="learning rate")
parser.add_argument('--sample_every_iterations', default=1000, type=int, help="sample SR images for every number of iterations")
parser.add_argument('--save_folder', default='Experiment', type=str, help="Folder to save your train or evaluation result")
parser.add_argument('--load_path', default=None, type=str, help="None or directory to pretrained model")
parser.add_argument('--data_path', default='images1024x1024/', type=str, help="directory to your training dataset")
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--input_image", default="test_images/16_128_lr_image", type=str, help="For evaluation, input folder included LR images")
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
timestep = (args.hr_size - args.lr_size) // args.stride
print(f"Total steps for {args.lr_size} to {args.hr_size}: {timestep}")

if args.hr_size == 512:
    model = Unet(
        dim = 32,                             # 32 ==> 512
        dim_mults = (1, 2, 4, 4, 8, 8, 16),   # 32 ==> 512
        channels=3,
        residual=False
    ).to(device)
elif args.hr_size == 128:
    model = Unet(
        dim=64,                     # 16 ==> 128
        dim_mults=(1, 2, 4, 8),     # 16 ==> 128
        channels=3,
        residual=False
    ).to(device)
else:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8, 16),
        channels=3,
        residual=False
    ).to(device)

dtls = DTLS(
    model,
    image_size = args.hr_size,
    stride = args.stride,
    timesteps = timestep,        # number of steps
    device=device,
).to(device)


trainer = Trainer(
    dtls,
    args.data_path,
    image_size = args.hr_size,
    train_batch_size = args.batch_size,
    train_lr = args.lr_rate,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 1,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    input_image = args.input_image,
    device = device,
    save_and_sample_every = args.sample_every_iterations
)

if args.mode == 'train':
    trainer.train()
elif args.mode == 'eval':
    trainer.evaluation()
