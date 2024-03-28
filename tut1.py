import bokeh
from bokeh.plotting import show
import matplotlib.pyplot as plt
import flowkit as fk

print(fk.__version__)
# help(fk.Sample)

fcs_path = '../fcs_data_V2/00001_17745_Tube_025.fcs'
sample = fk.Sample(fcs_path)
print(sample)
# Sample(v3.0, 17745_Tube_025.fcs, 16 channels, 75369 events)

print(sample.version)
print(sample.event_count)

# Retrieve all the metadata
print(sample.get_metadata())

# channel numbers indexed at 1, channel indices indexed at 0
print(sample.channels)

print("PnN labels:", sample.pnn_labels)
print("PnS labels:", sample.pns_labels)

# automatically identify fluorescent, scatter, and time channels
print(sample.fluoro_indices)
print(sample.scatter_indices)
print(sample.time_index)

# Retrieve sub-sampled indices
print(sample.subsample_indices)

# help(sample.plot_histogram)

p = sample.plot_histogram('FSC-H', source='raw', bins=256)
# show(p)

# help(sample.plot_channel)
f = sample.plot_channel('FSC-H', source='raw')
# plt.show()

# help(sample.plot_contour)
f = sample.plot_contour('FSC-H', 'SSC-H', fill=True, source='raw', plot_events=True)
# plt.show()

xform = fk.transforms.LogicleTransform('my_logicle', param_t=1024, param_w=0.5, param_m=4.5, param_a=0)
sample.apply_transform(xform)
p = sample.plot_scatter('FSC-A', 'FSC-H', source='raw')
# show(p)

# help(sample.plot_scatter_matrix)
spm = sample.plot_scatter_matrix(
    source='xform',
    channel_labels_or_numbers=['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W'],
    color_density=True
)
# show(spm)

# help(sample.get_events)
sample.get_events(source='raw')
channel_idx = sample.get_channel_index('FSC-H')
print(sample.get_channel_events(channel_idx, source='raw'))

# help(sample.export)

