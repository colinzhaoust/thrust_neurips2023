export model=t5-base

# base
for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
	echo $layer
	python ./easy_thrust_qa.py --task nq --model $model --target_layer $layer
done