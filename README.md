# MVHR-DP

Software Defect Prediction via Code Multi-View Hypergraph Representation Learning

We proposed an MVHR-DP method. The method constructs a multi-view hypergraph structure based on different perspectives of code features, fuses the multi-view information into a hypergraph with adjustable dimensions for hyperedge convolution operation, and performs deeper correlation feature extraction on the code.

# Prepare for dataset

Before training MVHR-DP, we have to provide the features of the hypergraph nodes. Thus, three types of node metrics are introuduced as node features:

- Traditional Static Code Metric:*20 manually designed metrics can be generated by open tool  [Understand]( https://scitools.com) (Process-Binary.csv).

- Complex Network Metric:*widely used in social network and included 17 metrics (Process-Metric.csv).

- Network Embedding Metric: use the [ProNE](https://github.com/THUDM/ProNE) implementation to generate the network embedding file (Process-Vector.csv).

- Class Dependency Network: use to generate Complex Network Metric and code view supplementary information can be acquired via the publicly available API [Dependencyfinder](https://depfind.sourceforge.io/) (dependencies_edges.csv).

We conduct experiments on ten open-source Java projects (a total of 29 versions). The metrics of the ten projects are saved in [data](https://github.com/insoft-lab/MVHR-DP/blob/main/data)

# Build running environment

- `Install required packages.`

```
pip install -r requirements.txt
```

# Train and test

If perform within-project defect prediction, run_WPDP.py. 

If perform cross-version defect prediction, run_CVDP.py.

If perform cross-project defect prediction, run_CPDP.py. 

```
python run_WPDP.py

python run_CVDP.py

python run_CPDP.py
```

# Check experimental result

- Please check the generated experimental results folder [result_WPDP](https://github.com/insoft-lab/MVHR-DP/tree/main/results_WPDP) , [result_CVDP](https://github.com/insoft-lab/MVHR-DP/tree/main/results_CVDP) and [result_CPDP](https://github.com/insoft-lab/MVHR-DP/tree/main/results_CPDP)
- Comparison results with all baseline methods [WPDP](https://github.com/insoft-lab/MVHR-DP/blob/main/results_comparison/WPDP.pdf), [CVDP](https://github.com/insoft-lab/MVHR-DP/blob/main/results_comparison/CVDP.pdf) and [CPDP](https://github.com/insoft-lab/MVHR-DP/blob/main/results_comparison/CPDP.pdf)

