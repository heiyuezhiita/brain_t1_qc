# 2025.03.21 hook_tool for model. In general, save the extracted features in "self.features"

# 2025.03.21, hook for ResNet, get last FC layer input
class ResNetHookForLastFcLayer(object):
    def __init__(self, net):
        # save hooks in this list
        self.hooks = []
        # save output feature maps in this dic
        self.features = dict()

        self.hooks.append(
            net.conv_seg[2].register_forward_hook(self._build_hook("last_fc_input")))

    def _build_hook(self, idx):
        # hook must have those three input args
        def hook(module, module_input, module_output):
            # module_input in last_fc_input for ResNet is a tuple...
            self.features[idx] = module_input[0]

        return hook

    def remove(self):
        for hook in self.hooks:
            hook.remove()
   






