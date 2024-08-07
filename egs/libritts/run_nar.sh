export PYTHONPATH=$PYTHONPATH:/workspace/icefall_llm
export PYTHONPATH=$PYTHONPATH:/workspace/vall-e

pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html

pip install -r /workspace/Amphion/requirements.txt
pip install phonemizer pypinyin sentencepiece kaldialign matplotlib h5py

apt-get update && apt-get -y install festival espeak-ng mbrola

world_size=8
exp_dir=exp/valle

## Train NAR model
# cd ${exp_dir}
# ln -s ${exp_dir}/best-valid-loss.pt epoch-99.pt  # --start-epoch 3=2+1
# cd -
python3 bin/trainer.py --max-duration 160 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 2 \
      --num-buckets 6 --dtype "float32" --save-every-n 1000 --valid-interval 2000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.03 --warmup-steps 200 --average-period 0 \
      --num-epochs 40 --start-epoch 100 --start-batch 0 --accumulate-grad-steps 2 \
      --exp-dir ${exp_dir} --world-size ${world_size}