#!/bin/bash

### conda activate poincare
export wd='/Chromatin/genome-poincare-embeddings'

export nstart=1
export nend=25 ### 20 to 30 suffices

while [ $nstart -le $nend ]; do

cd ${wd}/LG/
python stochastic_graph.py  ### yield random_graph
echo "Graph Sampling done."

cd ${wd}/

if [ $nstart -eq 1 ]; then
./first-train-hic.sh
else
./restart-train-hic.sh ${nstart}
fi

mv keys.txt ${wd}/LG/keys.$[$nstart -1].txt
mv pe.coors.txt ${wd}/LG/pe.coors.$[$nstart -1].txt

export nstart=$[ $nstart + 1 ]
done
