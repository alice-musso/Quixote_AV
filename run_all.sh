#!/bin/bash

mkdir -p results/outputs
mkdir -p results/inference
BASE_DIR=$(dirname "$0")
BASE_DIR=$(cd "$BASE_DIR" && pwd)

TRAIN_DIR="$BASE_DIR/corpus/training"
TEST_DIR="$BASE_DIR/corpus/test"

MODEL_TYPE=(
#"svm"
"lr"
)

AUTHORS=(
    "Mateo Alemán"
    #"Cervantes"
    #"Lope de Vega"
    "Agustín de Rojas Villandrando"
    "Alonso de Castillo Solórzano"
    "Guillén de Castro"
    "Juan Ruiz de Alarcón y Mendoza"
    #"Pasamonte"
    "Pérez de Hita"
    "Quevedo"
    #"Tirso de Molina"
    #"Navarrete"
)

for AUTHOR in "${AUTHORS[@]}"; do
    NAME_PATH=$(echo "$AUTHOR" | iconv -t ascii//TRANSLIT | tr ' ' '_')
    AUTHOR_NORMALIZED=$(echo "$AUTHOR" | iconv -t ascii//TRANSLIT)
    echo ">>> Running inference for: $AUTHOR"

    for MODEL in "${MODEL_TYPE[@]}"; do
        cd "$BASE_DIR/src"

        python -m inference \
            --train-dir="$TRAIN_DIR" \
            --test-dir="$TEST_DIR" \
            --positive-author="$AUTHOR_NORMALIZED" \
            --classifier-type="$MODEL" \
            --results-inference="$BASE_DIR/results/inference/dummy.csv" \
            > "$BASE_DIR/results/outputs/output_${NAME_PATH}_${MODEL}.txt" 2>&1
    done
done
