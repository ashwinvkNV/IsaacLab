# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from leapp import annotate
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.utils.warp.proxy_array import ProxyArray

from .leapp_semantics import LeappTensorSemantics, resolve_leapp_element_names, select_element_names


def _normalize_index(index: Any) -> Any:
    """Return a hashable representation of a tensor index."""
    if isinstance(index, slice):
        return ("slice", index.start, index.stop, index.step)
    if index is Ellipsis:
        return ("ellipsis",)
    if index is None:
        return ("none",)
    if isinstance(index, torch.Tensor):
        return (
            "tensor",
            tuple(index.detach().cpu().reshape(-1).tolist()),
            str(index.dtype),
            tuple(index.shape),
        )
    if isinstance(index, list | tuple):
        return tuple(_normalize_index(value) for value in index)
    if isinstance(index, int):
        return ("int", int(index))
    return ("value", index)


def _is_full_slice(index: Any) -> bool:
    """Return whether *index* selects the full axis."""
    return isinstance(index, slice) and index == slice(None)


def _element_axis_index(selection: Any) -> Any | None:
    """Return the tensor index that applies to the first semantic element axis.

    LEAPP tensor semantics describe the non-batch axes. Isaac Lab state tensors
    are batch-first, so ``tensor[:, joint_ids]`` maps ``joint_ids`` to element
    names axis 0.
    """
    if selection is None:
        return None
    if not isinstance(selection, tuple):
        return selection
    if len(selection) < 2:
        return None
    return selection[1]


def _select_semantic_element_names(element_names: list | None, selection: Any) -> list | None:
    """Select semantic element names for an indexed tensor view."""
    if element_names is None:
        return None
    axis_index = _element_axis_index(selection)
    if axis_index is None or _is_full_slice(axis_index):
        return element_names

    if all(isinstance(name, str) for name in element_names):
        return select_element_names(element_names, axis_index)
    if all(isinstance(axis_names, list) for axis_names in element_names):
        selected = list(element_names)
        selected[0] = select_element_names(element_names[0], axis_index)
        return selected
    return element_names


class LazyTracedTensor:
    """Lazily annotate a state tensor when its final indexed value is known."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        input_name: str,
        semantics_meta: LeappTensorSemantics,
        real_data: Any,
        entity_name: str,
        property_name: str,
        task_name: str,
        cache: dict,
    ) -> None:
        self._tensor = tensor
        self._input_name = input_name
        self._semantics_meta = semantics_meta
        self._real_data = real_data
        self._entity_name = entity_name
        self._property_name = property_name
        self._task_name = task_name
        self._cache = cache

    def _annotate(self, selection: Any = None) -> torch.Tensor:
        cache_key = (
            id(self._real_data),
            self._property_name,
            _normalize_index(selection),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        ref = self._tensor if selection is None else self._tensor[selection]
        element_names = _select_semantic_element_names(
            resolve_leapp_element_names(self._semantics_meta, self._real_data),
            selection,
        )
        sem = TensorSemantics(
            name=self._input_name,
            ref=ref,
            kind=self._semantics_meta.kind,
            element_names=element_names,
            extra=build_state_connection(self._entity_name, self._property_name),
        )
        annotated = annotate.input_tensors(self._task_name, sem)
        self._cache[cache_key] = annotated
        return annotated

    def __getitem__(self, selection: Any) -> torch.Tensor:
        return self._annotate(selection)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._annotate(), name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Materialize lazy tensors when they participate in torch operations."""
        if kwargs is None:
            kwargs = {}

        def unwrap(value):
            if isinstance(value, cls):
                return value._annotate()
            if isinstance(value, tuple):
                return tuple(unwrap(item) for item in value)
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            if isinstance(value, dict):
                return {key: unwrap(item) for key, item in value.items()}
            return value

        return func(*unwrap(args), **unwrap(kwargs))


class TracedProxyArray(ProxyArray):
    _traced_array: LazyTracedTensor

    def __init__(
        self,
        proxy_array: ProxyArray,
        *,
        input_name: str,
        semantics_meta: LeappTensorSemantics,
        real_data: Any,
        entity_name: str,
        property_name: str,
        task_name: str,
        cache: dict,
    ) -> None:
        super().__init__(proxy_array.warp)
        astorch = super().torch
        traced = LazyTracedTensor(
            astorch,
            input_name=input_name,
            semantics_meta=semantics_meta,
            real_data=real_data,
            entity_name=entity_name,
            property_name=property_name,
            task_name=task_name,
            cache=cache,
        )
        object.__setattr__(self, "_traced_array", traced)

    @property
    def torch(self) -> LazyTracedTensor:
        return self._traced_array

    @property
    def warp(self) -> Any:
        raise AttributeError("warp arrays are not supported for leapp export")


def ensure_env_spec_id(env, fallback_task_name: str = "policy") -> str:
    """Return ``env.unwrapped.spec.id``, creating a fallback spec when needed."""
    spec = getattr(env.unwrapped, "spec", None)
    if spec is None:
        env.unwrapped.spec = SimpleNamespace(id=fallback_task_name)
        return fallback_task_name

    task_name = getattr(spec, "id", None)
    if task_name is None:
        spec.id = fallback_task_name
        return fallback_task_name

    return task_name


# ══════════════════════════════════════════════════════════════════
# Connection Builders
# ══════════════════════════════════════════════════════════════════


def build_state_connection(entity_name: str, property_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for a state property."""
    return {"isaaclab_connection": f"state:{entity_name}:{property_name}"}


def build_command_connection(command_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for a command term."""
    return {"isaaclab_connection": f"command:{command_name}"}


def build_observation_connection(group_name: str, term_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for an observation term."""
    return {"isaaclab_connection": f"observation:{group_name}:{term_name}"}


def build_write_connection(entity_name: str, method_name: str) -> dict[str, str]:
    """Return a compact deployment connection string for an articulation write target."""
    return {"isaaclab_connection": f"write:{entity_name}:{method_name}"}
