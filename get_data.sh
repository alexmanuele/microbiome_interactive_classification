#!/bin/bash
mkdir datasets/
mkdir datasets/IBD_Gevers
mkdir datasets/plant_v_animal
mkdir datasets/usa_vs_malawi
#Gevers 2014, IBD vs control
wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/gevers/gg/taxatable.txt -O datasets/IBD_Gevers/taxa_gg.txt
wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/gevers/gg/otutable.txt -O datasets/IBD_Gevers/otu_gg.txt
wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/gevers/refseq/taxatable.txt -O datasets/IBD_Gevers/taxa_refseq.txt
wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/gevers/refseq/otutable.txt -O datasets/IBD_Gevers/otu_refseq.txt
wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/gevers/task-ileum.txt -O datasets/IBD_Gevers/task.txt
#One task per dataset for now, to allow consistent file naming.
#wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/gevers/task-rectum.txt -O datasets/IBD_Gevers/task.txt

# David 2014, animal vs plnat diet stool
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/david/gg/otutable.txt?raw=true -O datasets/plant_v_animal/otu_gg.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/david/gg/taxatable.txt?raw=true -O datasets/plant_v_animal/taxa_gg.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/david/refseq/otutable.txt?raw=true -O datasets/plant_v_animal/otu_refseq.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/david/refseq/taxatable.txt?raw=true -O datasets/plant_v_animal/taxa_refseq.txt
wget https://raw.githubusercontent.com/knights-lab/MLRepo/master/datasets/david/task.txt -O datasets/plant_v_animal/task.txt

# usa vs malawi, Yatsunenko 2013
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/yatsunenko/gg/otutable.txt?raw=true -O datasets/usa_vs_malawi/otu_gg.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/yatsunenko/gg/taxatable.txt?raw=true -O datasets/usa_vs_malawi/taxa_gg.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/yatsunenko/refseq/otutable.txt?raw=true -O datasets/usa_vs_malawi/otu_refseq.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/yatsunenko/refseq/taxatable.txt?raw=true -O datasets/usa_vs_malawi/taxa_refseq.txt
wget https://github.com/knights-lab/MLRepo/blob/master/datasets/yatsunenko/task-usa-malawi.txt?raw=true -O datasets/usa_vs_malawi/task.txt
