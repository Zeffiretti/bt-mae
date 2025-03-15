pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install timm==0.3.2 # required by the original code
pip install tensorboard
pip install numpy==1.23.5 # higher version of numpy causes error module 'numpy' has no attribute 'float'.


## Commands below will apply timm modification for compatibility with PyTorch 1.8+
# See https://github.com/huggingface/pytorch-image-models/issues/420#issuecomment-776459842
# timm_path=$(python -c "import timm; print(timm.__path__[0])")
timm_path=$(pip show timm | grep "Location" | awk '{print $2}')/timm

helpers_file="$timm_path/models/layers/helpers.py"

target_line="from torch._six import container_abcs"

# check if the helpers.py file exists
if [[ -f "$helpers_file" ]]; then
    if sed -n '6p' "$helpers_file" | grep -q "$target_line"; then
        cp "$helpers_file" "$helpers_file.bak"
        
        awk 'NR==6 {
            print "import torch";
            print "TORCH_MAJOR = int(torch.__version__.split(\".\")[0])";
            print "TORCH_MINOR = int(torch.__version__.split(\".\")[1])";
            print "";
            print "if TORCH_MAJOR == 1 and TORCH_MINOR < 8:";
            print "    from torch._six import container_abcs";
            print "else:";
            print "    import collections.abc as container_abcs";
            next;
        }1' "$helpers_file.bak" > "$helpers_file"
        
        echo "Modification applied successfully. Backup created as helpers.py.bak"
    else
        echo "No modification needed. The target line is not found at line 6."
    fi
else
    echo "helpers.py file not found!"
fi
