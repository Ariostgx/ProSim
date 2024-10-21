
r"""
Modified from Habitat codebase

Import the global registry object using

.. code:: py

Various decorators for registry different kind of classes with unique keys

-   Register a model: ``@registry.register_model``
-   Register a metric: ``@registry.register_metric``
-   Register a dataset: ``@registry.register_dataset``

"""

import collections
from typing import Any, Callable, DefaultDict, Optional, Type

from torch import nn
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule
from torchmetrics import Metric

class Registry():
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(
                    to_register, assert_type
                )
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "dataset", to_register, name, assert_type=Dataset
        )

    @classmethod
    def register_metric(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "metric", to_register, name, assert_type=Metric
        )

    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "model", to_register, name, assert_type=LightningModule
        )

    @classmethod
    def register_scene_encoder(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "scene_encoder", to_register, name, assert_type=nn.Module
        )

    @classmethod
    def register_prompt_encoder(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "prompt_encoder", to_register, name, assert_type=nn.Module
        )

    @classmethod
    def register_decoder(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "decoder", to_register, name, assert_type=nn.Module
        )

    @classmethod
    def register_hist_encoder(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "hist_encoder", to_register, name, assert_type=nn.Module
        )

    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl(
            "policy", to_register, name, assert_type=nn.Module
        )

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_dataset(cls, name: str) -> Type[Dataset]:
        return cls._get_impl("dataset", name)

    @classmethod
    def get_metric(cls, name: str) -> Type[Metric]:
        return cls._get_impl("metric", name)

    @classmethod
    def get_model(cls, name: str) -> Type[LightningModule]:
        return cls._get_impl("model", name)

    @classmethod
    def get_scene_encoder(cls, name: str) -> Type[nn.Module]:
        return cls._get_impl("scene_encoder", name)
    
    @classmethod
    def get_prompt_encoder(cls, name: str) -> Type[nn.Module]:
        return cls._get_impl("prompt_encoder", name)
    
    @classmethod
    def get_hist_encoder(cls, name: str) -> Type[nn.Module]:
        return cls._get_impl("hist_encoder", name)

    @classmethod
    def get_policy(cls, name: str) -> Type[nn.Module]:
        return cls._get_impl("policy", name)
    
    @classmethod
    def get_decoder(cls, name: str) -> Type[nn.Module]:
        return cls._get_impl("decoder", name)


registry = Registry()
