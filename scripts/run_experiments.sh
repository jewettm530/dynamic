#!/bin/bash
set -e  # exit on error

# Make sure the echonet package is installed
python3 -c "import echonet" || { echo "echonet module not found. Run 'pip install -e .' first."; exit 1; }

for pretrained in True False
do
    for model in r2plus1d_18 r3d_18 mc3_18
    do
        for frames in 96 64 32 16 8 4 1
        do
            batch=$((256 / frames))
            batch=$(( batch > 16 ? 16 : batch ))
            if [ $batch -lt 1 ]; then batch=1; fi

            cmd="import echonet; echonet.utils.video.run(modelname=\"${model}\", frames=${frames}, period=1, pretrained=${pretrained}, batch_size=${batch})"
            echo "Running: $cmd"
            python3 -c "${cmd}"
        done
        for period in 2 4 6 8
        do
            batch=$((256 / 64 * period))
            batch=$(( batch > 16 ? 16 : batch ))
            if [ $batch -lt 1 ]; then batch=1; fi

            cmd="import echonet; echonet.utils.video.run(modelname=\"${model}\", frames=$((64 / period)), period=${period}, pretrained=${pretrained}, batch_size=${batch})"
            echo "Running: $cmd"
            python3 -c "${cmd}"
        done
    done
done

period=2
pretrained=True
for model in r2plus1d_18 r3d_18 mc3_18
do
    cmd="import echonet; echonet.utils.video.run(modelname=\"${model}\", frames=$((64 / period)), period=${period}, pretrained=${pretrained}, run_test=True)"
    echo "Running: $cmd"
    python3 -c "${cmd}"
done

python3 -c "import echonet; echonet.utils.segmentation.run(modelname=\"deeplabv3_resnet50\", save_segmentation=True, pretrained=False)"

pretrained=True
model=r2plus1d_18
period=2
batch=$((256 / 64 * period))
batch=$(( batch > 16 ? 16 : batch ))
if [ $batch -lt 1 ]; then batch=1; fi
for patients in 16 32 64 128 256 512 1024 2048 4096 7460
do
    epochs=$(( 50 * (8192 / patients) ))
    if [ $epochs -lt 1 ]; then epochs=1; fi
    if [ $epochs -gt 200 ]; then epochs=200; fi
    cmd="import echonet; echonet.utils.video.run(modelname=\"${model}\", frames=$((64 / period)), period=${period}, pretrained=${pretrained}, batch_size=${batch}, num_epochs=${epochs}, output=\"output/training_size/video/${patients}\", n_train_patients=${patients})"
    echo "Running: $cmd"
    python3 -c "${cmd}"
    
    cmd="import echonet; echonet.utils.segmentation.run(modelname=\"deeplabv3_resnet50\", pretrained=False, num_epochs=${epochs}, output=\"output/training_size/segmentation/${patients}\", n_train_patients=${patients})"
    echo "Running: $cmd"
    python3 -c "${cmd}"
done