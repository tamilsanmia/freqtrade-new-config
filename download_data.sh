#!/bin/bash

# Define the timeframes you want to download
# timeframes=("5m" "15m" "30m" "1h" "4h" "1d")
timeframes=("1m" "5m" "15m" "30m" "1h" "4h" "1d")

# Set timerange and config
timerange="20221216-20251216"
config="user_data/binance_futures_Ichimoku_PairOptimized.json"

# Loop through timeframes and download data
for tf in "${timeframes[@]}"; do
    echo "📥 Downloading data for timeframe: $tf"

    docker run --rm \
        -v "$PWD/user_data:/freqtrade/user_data" \
        freqtradeorg/freqtrade:stable \
        download-data \
        --exchange binance \
        --config "$config" \
        --prepend \
        --timerange "$timerange" \
        --timeframe "$tf"
done