import torch
from collections import OrderedDict
def inflate_from_2d_model(state_dict_2d, state_dict_3d, skipped_keys=None, inflated_dim=2, is_mobilenet3d_v2=False):
    #TODO(yue)
    if is_mobilenet3d_v2:
        # TODO(DEBUG)
        pairs = [
            ["conv.0.0", "conv.0"],
            ["conv.0.1", "conv.1"],
            ["conv.1.0", "conv.3"],
            ["conv.1.1", "conv.4"],
            ["conv.2", "conv.6"],
            ["conv.3", "conv.7"],
            ["conv.1", "conv.3"],
            # ["conv.2", "conv.4"],
        ]

        old_keys_del = []
        new_keys_add = []
        for old_key in state_dict_2d.keys():
            if "features.0" not in old_key:
                for pair in pairs:
                    if pair[0] in old_key:
                        if pair[0] == "conv.2" and "features.1." in old_key:
                            new_key = old_key.replace(pair[0], "conv.4")
                        else:
                            new_key = old_key.replace(pair[0], pair[1])
                        old_keys_del.append(old_key)
                        new_keys_add.append(new_key)
                        break
        state_dict_2d_new = {}
        for key in state_dict_2d:
            if key not in old_keys_del:
                state_dict_2d_new[key] = state_dict_2d[key]

        for i in range(len(old_keys_del)):
            state_dict_2d_new[new_keys_add[i]] = state_dict_2d[old_keys_del[i]]

        state_dict_2d = state_dict_2d_new


    if skipped_keys is None:
        skipped_keys = []
    missed_keys = []
    new_keys = []
    for old_key in state_dict_2d.keys():
        if old_key not in state_dict_3d.keys():
            missed_keys.append(old_key)
    for new_key in state_dict_3d.keys():
        if new_key not in state_dict_2d.keys():
            new_keys.append(new_key)

    print("Missed tensors: {}".format(missed_keys))
    # print("New tensors: {}".format(new_keys))
    print("New tensors(except batch_tracked): {}".format([x for x in new_keys if "num_batches_tracked" not in x])) #TODO(yue)
    print("Following layers will be skipped: {}".format(skipped_keys))

    state_d = OrderedDict()
    unused_layers = [k for k in state_dict_2d.keys()]
    uninitialized_layers = [k for k in state_dict_3d.keys()]
    initialized_layers = []
    for key, value in state_dict_2d.items():
        skipped = False
        for skipped_key in skipped_keys:
            if skipped_key in key:
                skipped = True
                break
        if skipped:
            continue
        new_value = value
        # only inflated conv's weights
        if key in state_dict_3d:
            # TODO: a better way to identify conv layer?
            # if 'conv.weight' in key or \
            #         'conv1.weight' in key or 'conv2.weight' in key or 'conv3.weight' in key or \
            #         'downsample.0.weight' in key:
            if value.ndimension() == 4 and 'weight' in key:
                value = torch.unsqueeze(value, inflated_dim)
                # value.unsqueeze_(inflated_dim)
                repeated_dim = torch.ones(state_dict_3d[key].ndimension(), dtype=torch.int)
                repeated_dim[inflated_dim] = state_dict_3d[key].size(inflated_dim)
                new_value = value.repeat(repeated_dim.tolist())
            state_d[key] = new_value
            initialized_layers.append(key)
            uninitialized_layers.remove(key)
            unused_layers.remove(key)
    # print("Initialized layers: {}".format(initialized_layers))
    print("Initialized layers: {} vars".format(len(initialized_layers))) #TODO(yue)
    # print("Uninitialized layers: {}".format(uninitialized_layers))
    print("Uninitialized layers(except batch_tracked): {}".format([x for x in uninitialized_layers if "num_batches_tracked" not in x])) #TODO(yue)
    print("Unused layers: {}".format(unused_layers))
    return state_d