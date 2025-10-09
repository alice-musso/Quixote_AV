#!/bin/bash

mkdir -p results

BASE_DIR=$(dirname "$0")
TRAIN_DIR="$BASE_DIR/corpus/training"
TEST_DIR="$BASE_DIR/corpus/test"

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
    NAME_PATH=$(echo "$AUTHOR" | iconv -t ascii//TRANSLIT| tr ' ' '_')
    # il primo toglie gli accenti, il secondo il secondo mette gli underscore al posto dello spazio
    echo ">>> Running inference for: $AUTHOR"
    AUTHOR_NORMALIZED=$(echo "$AUTHOR" | iconv -t ascii//TRANSLIT)
    #no accenti



    PYTHONPATH=.:..:src python -m src.main_inference \
        --train-dir="$TRAIN_DIR" \
        --test-dir="$TEST_DIR" \
        --positive-author="$AUTHOR_NORMALIZED" \
        > "outputs/output_${NAME_PATH}.txt"

    if [ ! -s "outputs/output_${NAME_PATH}.txt" ]; then
        echo "Nessun documento trovato per $AUTHOR"
    fi
done

echo "File salvati in: outputs/"