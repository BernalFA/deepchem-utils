import deepchem as dc


FEATURIZERS = {
    "MolGraphConvFeaturizer": dc.feat.MolGraphConvFeaturizer(use_edges=True),
    "ConvMolFeaturizer": dc.feat.ConvMolFeaturizer(),
    "DMPNNFeaturizer": dc.feat.DMPNNFeaturizer(),
    "DummyFeaturizer": dc.feat.DummyFeaturizer(),
}


MODELS = {
    "GraphConvModel": dc.models.GraphConvModel,
    "MPNNModel": dc.models.torch_models.MPNNModel,
    "DMPNNModel": dc.models.torch_models.DMPNNModel,
    "GCNModel": dc.models.torch_models.GCNModel,
    "AttentiveFPModel": dc.models.torch_models.AttentiveFPModel,
    "GATModel": dc.models.torch_models.GATModel,
    "DAGModel": dc.models.DAGModel,
}
