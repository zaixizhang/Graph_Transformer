[ -z "${exp_name}" ] && exp_name="cora"
[ -z "${epoch}" ] && epoch="1000"
[ -z "${seed}" ] && seed="42"
[ -z "${arch}" ] && arch="--ffn_dim 128 --hidden_dim 128 --dropout_rate 0.5 --n_layers 6 --peak_lr 2e-4"
[ -z "${batch_size}" ] && batch_size="16"
[ -z "${data_augment}" ] && data_augment="4"
[ -z "${n_gpu}" ] && n_gpu="1"

max_epochs=$((epoch+1))
echo "=====================================ARGS======================================"
echo "max_epochs: ${max_epochs}"
echo "==============================================================================="


echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="/exps/$exp_name/$seed"
mkdir -p $default_root_dir

python main.py --seed $seed --batch_size $batch_size \
      --dataset_name $exp_name --epochs $epoch\
      $arch \
      --checkpoint_path $default_root_dir\
      --num_data_augment $data_augment

echo "=====================================EVAL======================================"

