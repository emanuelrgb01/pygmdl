#!/bin/bash

set -e

echo "--- GMDL Iris Dataset Demonstration ---"

# Step 1
echo -e "\n[1/5] Preparing Iris dataset..."

DATA_DIR="iris_data"
mkdir -p $DATA_DIR

FULL_CSV="$DATA_DIR/iris_with_header.csv"
SHUFFLED_CSV="$DATA_DIR/iris_shuffled.csv"
TRAIN_CSV="$DATA_DIR/iris_train.csv"
TEST_CSV="$DATA_DIR/iris_test.csv"
ONLINE_STREAM="$DATA_DIR/iris_online_stream.txt"

echo "Downloading Iris dataset..."
wget -q -O "$DATA_DIR/iris.data" https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

echo "sepal_length,sepal_width,petal_length,petal_width,species" > $FULL_CSV
cat "$DATA_DIR/iris.data" >> $FULL_CSV

tail -n +2 $FULL_CSV | shuf > $SHUFFLED_CSV

echo "sepal_length,sepal_width,petal_length,petal_width,species" > $SHUFFLED_CSV
tail -n +2 $FULL_CSV | shuf >> $SHUFFLED_CSV

head -n 106 $SHUFFLED_CSV > $TRAIN_CSV
{ head -n 1 $SHUFFLED_CSV; tail -n 45 $SHUFFLED_CSV; } > $TEST_CSV

echo "Dataset prepared successfully in '$DATA_DIR' directory."

# Step 2
echo -e "\n[2/5] Running GMDL in Batch Mode..."

python -m examples.run_batch \
    --training-file "$TRAIN_CSV" \
    --testing-file "$TEST_CSV" \
    --label-column "species" \
    --labels "Iris-setosa,Iris-versicolor,Iris-virginica" \
    --confusion-matrix

echo "Batch mode execution finished."


# Step 3
echo -e "\n[3/5] Preparing data stream for Online Mode..."

{
    awk -F',' 'NR>1 {print "<Training>"; print $0}' "$TRAIN_CSV"
    awk -F',' 'NR>1 {print "<Test>"; print $0}' "$TEST_CSV"

} > $ONLINE_STREAM

echo "Online stream created at '$ONLINE_STREAM'."

# Step 4
echo -e "\n[4/5] Running GMDL in Online Mode..."
echo "Predictions will be printed below:"

python -m examples.run_online \
    --labels "Iris-setosa,Iris-versicolor,Iris-virginica" \
    --dimension 4 \
    --confusion-matrix \
    --quiet < $ONLINE_STREAM

echo "Online mode execution finished."

# Step 5
echo -e "\n[5/5] Cleanup..."
read -p "Do you want to delete the generated data directory ('$DATA_DIR')? [y/N] " REPLY
echo  

if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Cleaning up generated files..."
    rm -r "$DATA_DIR"
    echo "Directory '$DATA_DIR' deleted."
else
    echo "Cleanup skipped. Data is available in the '$DATA_DIR' directory."
fi

echo "Done."