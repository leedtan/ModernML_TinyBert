import torch.nn as nn
from transformers.modeling_bert import BertLayerNorm

from mixout import MixLinear, mixout_layer


def manipulate_model(
    model,
    encoder_temp,
    args,
):
    encoder_temp = getattr(model, args.model_type)
    frozen_layers = args.frozen_layers
    mix_layers = args.mixout_layers
    finetune_layers = args.finetune_layers
    re_layers = args.reinit_layers
    first_frozen_index = 1
    first_mixout_index = 1 + frozen_layers * 12
    first_finetune_index = 1 + (frozen_layers + mix_layers) * 12
    12 * (12 - re_layers) + 1
    assert (12 * (12 - re_layers) + 1) == (
        1 + (frozen_layers + mix_layers + finetune_layers) * 12
    )
    layer_itr = 0
    mix_counter = 0
    for sup_module in list(model.modules()):
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Linear):
                if layer_itr == 0:
                    module.weight.data.normal_(
                        mean=0.0, std=encoder_temp.config.initializer_range
                    )
                # Frozen
                if layer_itr >= first_frozen_index and layer_itr < first_mixout_index:
                    for param in module.parameters():
                        param.requires_grad = False
                # Mixout
                if layer_itr >= first_mixout_index and layer_itr < first_finetune_index:
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    mix_depth = (layer_itr - 1) // (args.mixout_layers)
                    mix_depth = mix_depth / float(args.mixout_layers - 1)
                    mix_percent = (
                        mix_depth * args.mixout + (1 - mix_depth) * args.mixout_decay
                    )
                    if 1:
                        new_module = mixout_layer(
                            module,
                            mix_percent if args.mixout_decay > 0.0 else args.mixout,
                            args.device,
                            layer_mixout=args.layer_mixout,
                            frozen=True,
                            norm_flag=args.normalize,
                            name=str(mix_counter) + "_mix_layer",
                        )
                        mix_counter += 1
                    else:
                        new_module = MixLinear(
                            module.in_features,
                            module.out_features,
                            bias,
                            target_state_dict["weight"],
                            args.mixout,
                        )
                        new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)

                # Reinit (do nothing):

                layer_itr += 1
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    if args.reinit_layers > 0:
        for layer in encoder_temp.encoder.layer[-args.reinit_layers :]:
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(
                        mean=0.0, std=encoder_temp.config.initializer_range
                    )
                    module.weight.data.zero_()
                elif isinstance(module, BertLayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
