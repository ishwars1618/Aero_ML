"""Model type string -> constructor.

Lets a config name a model ("unet", "vit") instead of a training script hardcoding
a class, so switching architectures is a one-line edit to the config file.

Each entry is a lambda taking a DataConfig and returning a model on the CPU, ready
to train -- the caller does its own .to(device). "Ready to train" includes weight
initialization, which differs per architecture and is not optional for every model:

  unet -- nothing to do. StackedHourglass is built from stock nn.Conv2d /
          nn.BatchNorm2d / nn.ConvTranspose2d, each of which initializes itself in
          its own constructor. (The training script also defines a weights_init of
          its own, marked "Dont USE"; it is a disabled experiment, not a dependency.)
  vit  -- vit_model.weights_init is REQUIRED. VisionTransformer declares its
          attention matrices as nn.Parameter(torch.empty(...)), which allocates
          memory without writing to it, and nothing else ever fills them. Skipping
          this trains on whatever bytes were in that memory -- observed as a
          non-finite forward pass on every build under
          torch.use_deterministic_algorithms(True).

Architecture hyperparameters that are not in DataConfig (input channels, the ViT's
head and layer counts) stay here, next to the class they belong to.
"""

from unet_model import StackedHourglass
from vit_model import VisionTransformer, weights_init as vit_weights_init

IN_CHANNELS = 3


def _initialized(model, init_fn):
    """model.apply(init_fn), returning the model so registry entries stay one-liners."""
    model.apply(init_fn)
    return model


MODELS = {
    "unet": lambda config: StackedHourglass(
        IN_CHANNELS,
        config.num_landmarks,
    ),
    "vit": lambda config: _initialized(
        VisionTransformer(
            config.model_image_size,
            IN_CHANNELS,
            config.num_landmarks,
            4,   # numheads
            3,   # numlayers
        ),
        vit_weights_init,
    ),
}


def build_model(config, model=None):
    """Instantiate and initialize the model named by config.model (or by `model`).

    Raises on an unknown name rather than falling back to a default, so a typo in
    the config surfaces here instead of silently training the wrong network.
    """
    name = model if model is not None else config.model
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; known models: {sorted(MODELS)}")
    return MODELS[name](config)
