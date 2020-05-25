# prepare directories
cd ..
mkdir build && cd build

# installing apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# cleanup
cd ../..
rm -rf build
