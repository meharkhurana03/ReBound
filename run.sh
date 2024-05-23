#!/bin/bash

# Activate your Python environment if necessary

PYTHON_INTERPRETER_1="pipenv run python"
PYTHON_INTERPRETER="python"
SCRIPT_PATH="predict.py"
CDAL_PATH="CDAL_CS.py"
REBOUND_SCRIPT_PATH="/home/alive/Desktop/dhathri/ReBound/src/bash_scripts/mask_lct_2.py"
INPUT_FILES="trial"  
BUDGET=10  
OUTPUT_PATH="trial_out"
FEATURES="trial_out/averaged_softmax"  
NUM_CLASSES=23  
REBOUND_INPUT_PATH="$OUTPUT_PATH/images"
REBOUND_OUTPUT_PATH="$OUTPUT_PATH/images_out"

# Run the first script to generate the text file
$PYTHON_INTERPRETER $SCRIPT_PATH \
   --input "$INPUT_FILES" \
   --num_classes "$NUM_CLASSES"\
   --model "deeplabv3plus_mobilenet"\
   --output_stride 16\
   --gpu_id "0"\
   --save_val_results_to "$OUTPUT_PATH"

# Run the second script to process the text file and move the files to the specified directory
$PYTHON_INTERPRETER $CDAL_PATH \
   --budget "$BUDGET"\
   --file_path "$OUTPUT_PATH"\
   --feature "$FEATURES"\
   --nc "$NUM_CLASSES"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH/images"

mkdir -p "$OUTPUT_PATH/images_out"

# Copy and sort image files
index=0
while IFS= read -r image_name; do
    image_path="$INPUT_FILES/${image_name/_averaged_softmax}.jpg"

    if [ -f "$image_path" ]; then
       cp "$image_path" "$OUTPUT_PATH/images/$index.jpg"
       echo "Copied: $image_name to $OUTPUT_PATH/images/$index.jpg"
       ((index++))
    else
       echo "Image not found: $image_name"
    fi
done < "trial_outselected.txt"

# Activate virtual environment
conda activate annotation

# Run the ReBound script
$PYTHON_INTERPRETER_1 $REBOUND_SCRIPT_PATH -f "$REBOUND_INPUT_PATH" -o "$REBOUND_OUTPUT_PATH"

echo "SCRIPT COMPLETED"

