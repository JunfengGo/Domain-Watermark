

for var in {1}
do
 	CUDA_VISIBLE_DEVICES=2,3 python -u dw.py --retrain_max_epoch 150  --backdoor_poisoning --retrain_from_init --val_max_epoch 150  --epochs 150 --sources_train 1000  --defend_features_only --disable_adaptive_attack --vnet ResNet34 --net ResNet18 --threatmodel random-subset   --poison_path './poisons/poison_resnet18_mobile.pth' --budget 0.1 --save 'numpy' --coe 0.3 
	
       	
# 	#rm ./saved_test_imgs/*
done