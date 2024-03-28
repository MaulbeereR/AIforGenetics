import bokeh
from bokeh.plotting import show
import matplotlib.pyplot as plt
import numpy as np

import flowkit as fk

fcs_path = '../fcs_data_V2/00001_17745_Tube_025.fcs'
sample = fk.Sample(fcs_path)
print(sample.channels)

# Plot Untransformed
p = sample.plot_scatter(12, 10, source='raw', subsample=True)
show(p)

# Plot LogicleTransform
logicle_xform = fk.transforms.LogicleTransform('logicle', param_t=262144, param_w=0.5, param_m=4.5, param_a=0)
sample.apply_transform(logicle_xform)
p = sample.plot_scatter(12, 10, source='xform', subsample=True)
show(p)
# help(fk.transforms.LogicleTransform)

# Plot AsinhTransform
asinh_xform = fk.transforms.AsinhTransform('asinh', param_t=262144, param_m=4.0, param_a=0.0)
sample.apply_transform(asinh_xform)
p = sample.plot_scatter(12, 10, source='xform', subsample=True)
show(p)

# Plot WSPBiexTransform
biex_xform = fk.transforms.WSPBiexTransform('biex', max_value=262144.000029, positive=4.418540, width=-10, negative=0)
sample.apply_transform(biex_xform)
p = sample.plot_scatter(12, 10, source='xform', subsample=True)
show(p)

# help(fk.Matrix)
detectors = [sample.pnn_labels[i] for i in sample.fluoro_indices]
print(detectors)

# Apply the matrix from the 'spill' keyword and re-plot
sample.apply_compensation(sample.metadata['spill'])
fig = sample.plot_scatter(5, 8, source='xform', subsample=True)
show(fig)

comp_mat = sample.compensation
print(comp_mat.as_dataframe())
