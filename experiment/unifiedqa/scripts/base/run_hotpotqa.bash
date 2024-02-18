export model=allenai/unifiedqa-v2-t5-base-1251000

# base
for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
	echo $layer
	CUDA_VISIBLE_DEVICES=4 python ./easy_thrust_qa.py --task hotpotqa --model $model --target_layer $layer
done