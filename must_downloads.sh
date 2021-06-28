#!/bin/sh
echo "Downloading and copying the necessary model files"
BenchDIR=$(pwd)
echo ${BenchDIR}

echo "Downloading AlexNet Model Files"
cd ${BenchDIR}/VPR_Techniques/AlexNet_VPR/alexnet/
wget https://surfdrive.surf.nl/files/index.php/s/I6u78YACY3MEUeC/download
unzip download

echo "Downloading CALC Model Files"
cd  ${BenchDIR}/VPR_Techniques/CALC/model/
wget http://udel.edu/~nmerrill/calc.tar.gz
tar -xzvf calc.tar.gz calc_model/calc.caffemodel --strip-components=1

echo "Downloading AMOSNet Model Files"
cd ${BenchDIR}/VPR_Techniques/AmosNet/
wget https://goo.gl/6xtjwD
unzip 6xtjwD AmosNet.caffemodel

echo "Downloading HybridNet Model Files"
cd ${BenchDIR}/VPR_Techniques/HybridNet/
wget https://goo.gl/kF6nQh
unzip kF6nQh HybridNet.caffemodel

echo "Downloading NetVLAD Model Files"
cd ${BenchDIR}/VPR_Techniques/NetVLAD/checkpoints/
wget http://rpg.ifi.uzh.ch/datasets/netvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip
unzip -j vd16_pitts30k_conv5_3_vlad_preL2_intra_white.zip

echo "Downloading RegionVLAD Model Files"
cd ${BenchDIR}/VPR_Techniques/RegionVLAD/AlexnetPlaces365/
wget http://places2.csail.mit.edu/models_places365/alexnet_places365.caffemodel

echo "Downloading Variation Quantified Datasets"
cd ${BenchDIR}/
wget https://surfdrive.surf.nl/files/index.php/s/65AMTSuxxo2siJT/download
unzip download

echo "Note: I have not copied ALL the VPR datasets and the precomputed matching data, because they are not needed to run the first execution of VPR-Bench. For now, your VPR-Bench only has the Corridor dataset for traditional VPR evaluation of 8 VPR-techniques and the three variation quantified datasets for invariance analysis. See the main README.md or 'must_downloads.txt' for links to the other VPR datasets and the matching data."
