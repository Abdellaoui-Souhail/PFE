import argparse
import inspect
import numpy as np

from . import fdb as fdb
from .unet import UNetModel

def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=512,
        num_channels=32,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="8,4",
        dropout=0.0,
        learn_sigma=False,
        class_cond=False,
        diffusion_steps=100,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        undersampling_rate=2,
        data_type="multicoil",
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    use_checkpoint,
    use_scale_shift_norm,
    undersampling_rate,
    data_type,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        undersampling_rate=undersampling_rate,
        image_size=image_size,
        data_type=data_type,
    )
    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 320:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 384:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=2,
        model_channels=num_channels,
        out_channels=(2 if not learn_sigma else 8),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    undersampling_rate=2,
    image_size=256,
    data_type="singlecoil",
):

    return fdb.DiffusionBridge(
        steps=steps,
        undersampling_rate=undersampling_rate,
        image_size=image_size,
        data_type=data_type,
    )

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
