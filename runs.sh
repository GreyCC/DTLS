#training
python main.py --mode train --hr_size 512 --lr_size 32 --stride 16 --train_steps 100000 --save_folder '32_512_s16' --data_path 'your_dataset_directory' --batch_size 2

#Evaluation
python main.py --mode eval --hr_size 512 --lr_size 32 --stride 16 --load_path '32_512_s16/model.pt' --save_folder '32_512_s16_results' --input_image test_images/32_512_lr_image

