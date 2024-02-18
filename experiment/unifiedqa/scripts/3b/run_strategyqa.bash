export model=allenai/unifiedqa-v2-t5-3b-1251000

# base
for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
	echo $layer
	CUDA_VISIBLE_DEVICES=5 python ./easy_thrust_exp.py --task strategyqa --model $model --target_layer $layer
done