export model=t5-3b

# base
for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
	echo $layer
	python ./easy_thrust_qa.py --task triviaqa --model $model --target_layer $layer
done