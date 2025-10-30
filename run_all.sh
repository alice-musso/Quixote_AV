#!/bin/bash

mkdir -p results/outputs
mkdir -p results/inference
mkdir -p results/loo

BASE_DIR=$(dirname "$0")
TRAIN_DIR="$BASE_DIR/corpus/training"
TEST_DIR="$BASE_DIR/corpus/test"

MODEL_TYPE=(
"svm"
"lr"
)

AUTHORS=(
    "Mateo Alemán"
    "Cervantes"
    "Lope de Vega"
    #"Agustín de Rojas Villandrando"
    "Alonso de Castillo Solórzano"
    "Guillén de Castro"
    #"Juan Ruiz de Alarcón y Mendoza"
    #"Pasamonte"
    #"Pérez de Hita"
    #"Quevedo"
    #"Tirso de Molina"
)

for AUTHOR in "${AUTHORS[@]}"; do
    NAME_PATH=$(echo "$AUTHOR" | iconv -t ascii//TRANSLIT | tr ' ' '_')
    AUTHOR_NORMALIZED=$(echo "$AUTHOR" | iconv -t ascii//TRANSLIT)
    echo ">>> Running inference for: $AUTHOR"

    for MODEL in "${MODEL_TYPE[@]}"; do
        PYTHONPATH=.:..:src python -m src.main_inference \
            --train-dir="$TRAIN_DIR" \
            --test-dir="$TEST_DIR" \
            --positive-author="$AUTHOR_NORMALIZED" \
            --classifier-type="$MODEL" \
            --results-inference="results/inference/inference_results_${AUTHOR_NORMALIZED}_${MODEL}.csv" \
            --results-loo="results/loo/loo_results_${AUTHOR_NORMALIZED}_${MODEL}.csv" \
            > "results/outputs/output_${NAME_PATH}_${MODEL}.txt"

        if [ ! -s "results/outputs/output_${NAME_PATH}_${MODEL}.txt" ]; then
            echo "Nessun documento trovato per $AUTHOR"
        fi
    done
done

echo "File saved in results/inference, results/loo and results/outputs"
