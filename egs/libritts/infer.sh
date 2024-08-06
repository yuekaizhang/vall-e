export PYTHONPATH=$PYTHONPATH:/workspace/icefall_llm
export PYTHONPATH=$PYTHONPATH:/workspace/vall-e

install_flag=false
if [ "$install_flag" = true ]; then
    echo "Installing packages..."

    pip install k2==1.24.3.dev20230524+cuda11.8.torch2.0.1 -f https://k2-fsa.github.io/k2/cuda.html

    pip install -r /workspace/Amphion/requirements.txt
    pip install phonemizer pypinyin sentencepiece kaldialign matplotlib h5py

    apt-get update && apt-get -y install festival espeak-ng mbrola
else
    echo "Skipping installation."
fi
exp_dir=exp/valle
python3 bin/infer.py --output-dir demos \
    --checkpoint=${exp_dir}/best-valid-loss.pt \
    --text-prompts "KNOT one point one five miles per hour." \
    --audio-prompts ./prompts/8463_294825_000043_000000.wav \
    --text "To get up and running quickly just follow the steps below."

python3 bin/infer.py --output-dir demos \
        --top-k -1 --temperature 1.0 \
        --model-name valle --share-embedding true --norm-first true --add-prenet false \
        --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
        --text-prompts "" \
        --audio-prompts "" \
        --text ./libritts.txt \
        --checkpoint ${exp_dir}/best-valid-loss.pt