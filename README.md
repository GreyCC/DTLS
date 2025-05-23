# Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution - a Non-Denoising Model

This is the program code of our arXiv paper below.

Refer the paper from this website [https://arxiv.org/abs/2311.02358](https://arxiv.org/abs/2311.02358)

Please read the above paper and it can be cite as: 

Chun-Chuen Hui, Wan-Chi Siu, and Ngai-Fong Law, "Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution - A Non-Denoising Model," arXiv preprint arXiv:2311.02358, 2023. 


Or in Bibtex format:
```
@misc{hui2023domain,
      title={Domain Transfer in Latent Space (DTLS) Wins on Image Super-Resolution - a Non-Denoising Model}, 
      author={Chun-Chuen Hui and Wan-Chi Siu and Ngai-Fong Law},
      year={2023},
      eprint={2311.02358},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Dataset
To prepare FFHQ dataset, you can follow: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

## Training
Train the model by following the command lines below

**Face Super-resolution from 32 --> 512**
```
python main.py --mode train --hr_size 512 --lr_size 32 --stride 16 --train_steps 100000 --save_folder '32_512_s16' --data_path 'your_dataset_directory' --batch_size 2
```

**Face Super-resolution from 16 --> 128**
```
python main.py --mode train --hr_size 128 --lr_size 16 --stride 4 --train_steps 50000 --save_folder '16_128_s4' --data_path 'your_dataset_directory' --batch_size 32
```

## Recall
After the training you can run the following command to super-resolute face images for evaluation

For simple evaluation, you can download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1HKpawhbLtdTQzBAvD380rjKRwUCqIlDP?usp=sharing)

**Face Super-resolution from 32 --> 512**
```
python main.py --mode eval --hr_size 512 --lr_size 32 --stride 16 --load_path 'SR_32_512_s16.pt' --save_folder '32_512_s16_results' --input_image test_images/32_512_lr_image
```

**Face Super-resolution from 16 --> 128**
```
python main.py --mode eval --hr_size 128 --lr_size 16 --stride 4 --load_path 'SR_16_128_s4.pt' --save_folder '16_128_s4_results' --input_image test_images/16_128_lr_image
```

## Acknowledgements
This code is maninly built on [Cold Diffusion](https://github.com/arpitbansal297/Cold-Diffusion-Models)
