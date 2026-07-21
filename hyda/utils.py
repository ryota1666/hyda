import argparse
from importlib import import_module
import torch


def load_submodule_from_checkpoint(
    ckpt, submodule_name, existing_module=None, map_location="cuda:0"
):
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt, map_location=map_location)

    # 1. state_dict から該当サブモジュール（例: encoder.xxx -> xxx）の重みを抽出
    prefix = f"{submodule_name}."
    state_dict = {
        k[len(prefix) :]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(prefix)
    }

    # パターンA: 既にインスタンス（self.domain_encoderなど）が存在する場合は、重みだけをロードする
    if existing_module is not None:
        if state_dict:
            existing_module.load_state_dict(state_dict, strict=True)
            print(
                f"Successfully loaded {submodule_name} weights into existing module."
            )
        else:
            print(
                f"Warning: No weights found starting with '{prefix}' in state_dict."
            )
        return existing_module

    # パターンB: 従来通り、チェックポイント内のメタ情報から新規インスタンス化する
    hparams = ckpt.get("hyper_parameters", {})

    if "init_args" in hparams and submodule_name in hparams["init_args"]:
        submodule_cfg = hparams["init_args"][submodule_name]
    elif submodule_name in hparams:
        submodule_cfg = hparams[submodule_name]
    else:
        # メタ情報が存在しない場合は、全state_dictの読み込みを試みる
        raise KeyError(
            f"Could not find '{submodule_name}' in checkpoint hyper_parameters. Available keys: {list(hparams.keys())}"
        )

    module_name, class_name = submodule_cfg["class_path"].rsplit(".", 1)
    module_cls = getattr(import_module(module_name), class_name)
    module_kwargs = submodule_cfg.get("init_args", {})

    module_instance = module_cls(**module_kwargs)
    module_instance.load_state_dict(state_dict, strict=True)
    return module_instance


def namespace_to_dict(namespace):
    """Recursively converts an argparse.Namespace to a dictionary."""
    return {
        key: (
            namespace_to_dict(value)
            if isinstance(value, argparse.Namespace)
            else value
        )
        for key, value in vars(namespace).items()
    }