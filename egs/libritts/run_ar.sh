export PYTHONPATH=$PYTHONPATH:/workspace/icefall_llm
export PYTHONPATH=$PYTHONPATH:/workspace/vall-e

pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html

pip install -r /workspace/Amphion/requirements.txt
pip install phonemizer pypinyin sentencepiece kaldialign

apt-get update && apt-get install festival espeak-ng mbrola

world_size=8
exp_dir=exp/valle

## Train AR model
python3 bin/trainer.py --max-duration 320 --filter-min-duration 0.5 --filter-max-duration 14 --train-stage 1 \
      --num-buckets 6 --dtype "bfloat16" --save-every-n 1000 --valid-interval 2000 \
      --model-name valle --share-embedding true --norm-first true --add-prenet false \
      --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
      --base-lr 0.03 --warmup-steps 200 --average-period 0 \
      --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 1 \
      --exp-dir ${exp_dir} --world-size ${world_size}