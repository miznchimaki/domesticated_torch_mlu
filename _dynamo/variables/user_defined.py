import collections
import builtins
import contextlib
import dataclasses
import enum
import functools
import inspect
import itertools
import random
import sys
import threading
import types
import warnings
import weakref
from typing import Any, TYPE_CHECKING
from typing_extensions import is_typeddict
from collections.abc import Iterable, Sequence

import torch._dynamo.config
import torch.nn
import inspect
import threading
import types
from typing import Dict, List
import contextlib
import torch
from torch._dynamo import graph_break_hints, polyfills, variables
from torch._dynamo.exc import unimplemented
from torch._dynamo.guards import GuardBuilder, install_guard
from torch._dynamo import variables
from torch._dynamo.variables.base import (
    raise_type_error_exc,
    ValueMutationNew,
    VariableTracker,
)
from torch._dynamo.create_parameter_op import do_not_convert_to_tracable_parameter
from torch._dynamo.variables.user_defined import (
    is_standard_setattr,
    UserDefinedObjectVariable,
    is_standard_delattr,
    RandomVariable,
    is_forbidden_context_manager,
)
from torch._dynamo.utils import (
    check_constant_args,
    is_frozen_dataclass,
    is_namedtuple_cls,
    namedtuple_fields,
    proxy_args_kwargs,
    raise_args_mismatch,
    tensortype_to_dtype,
)
from torch._dynamo.source import (
    AttrSource,
    DataclassFieldsSource,
    GetItemSource,
)
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type
from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
from torch._dynamo.variables.dicts import ConstDictVariable, DefaultDictVariable
from torch._dynamo.variables.lists import SizeVariable
import functools
from ...utils import gorilla

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


# workaround for PYTORCH-12898
@gorilla.patch(
    torch._dynamo.variables.user_defined.UserDefinedObjectVariable,
    settings=gorilla.Settings(use_replace_references=True),
)
def call_method(
    self,
    tx: "InstructionTranslator",
    name: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> VariableTracker:
    # Modify by CAMBRICON
    # from . import CONSTANT_VARIABLE_NONE, ConstantVariable, UserMethodVariable
    from torch._dynamo.variables import (
        ConstantVariable,
        UserMethodVariable,
        CONSTANT_VARIABLE_NONE,
    )

    # end Modify by CAMBRICON

    method = self._maybe_get_baseclass_method(name)
    if method is not None:
        if method is object.__init__:
            return CONSTANT_VARIABLE_NONE

        if is_standard_setattr(method) or isinstance(self.value, threading.local):
            return self.method_setattr_standard(tx, *args, **kwargs)

        if is_standard_delattr(method):
            return self.method_setattr_standard(
                tx, args[0], variables.DeletedVariable()
            )

        # Add by CAMBRICON
        if method is set.__contains__ and len(args) == 1:
            return ConstantVariable.create(args[0].as_python_constant() in self.value)
        # end Add by CAMBRICON

        if method is object.__eq__ and len(args) == 1 and not kwargs:
            other = args[0]
            if not isinstance(other, UserDefinedObjectVariable):
                return variables.ConstantVariable.create(NotImplemented)

            # TODO(anijain2305) - Identity checking should already be a part
            # of the cmp_eq  polyfill function.
            return ConstantVariable.create(self.value is other.value)

        if torch._dynamo.config.enable_faithful_generator_behavior and isinstance(
            self.value, types.GeneratorType
        ):
            unimplemented(
                gb_type="call_method on generator",
                context=f"object={self.value}, method={name}, args={args}, kwargs={kwargs}",
                explanation="Detected a method call to a user-defined generator object. "
                "This is not fully supported.",
                hints=[
                    "Set `torch._dynamo.config.enable_faithful_generator_behavior = False`. Note that this "
                    "may cause silent incorrectness, since we will eagerly unpack generators instead of lazily "
                    "evaluating them.",
                ],
            )

        # check for methods implemented in C++
        if isinstance(method, types.FunctionType):
            source = self.source
            source_fn = None
            if source:
                source_fn = self.get_source_by_walking_mro(name)
            # TODO(jansel): add a guard to check for monkey patching?
            # Modify by CAMBRICON
            # from ..mutation_guard import unpatched_nn_module_init
            from torch._dynamo.mutation_guard import unpatched_nn_module_init

            # end Modify by CAMBRICON

            if method is torch.nn.Module.__init__:
                method = unpatched_nn_module_init
            return UserMethodVariable(
                method, self, source_fn=source_fn, source=source
            ).call_function(
                tx, args, kwargs
            )  # type: ignore[arg-type]

        if method is list.__len__ and self.source and not (args or kwargs):
            install_guard(self.source.make_guard(GuardBuilder.SEQUENCE_LENGTH))
            return ConstantVariable(len(self.value))  # type: ignore[arg-type]

    # Modify by CAMBRICON
    # return super().call_method(tx, name, args, kwargs)
    return super(UserDefinedObjectVariable, self).call_method(tx, name, args, kwargs)
    # end Modify by CAMBRICON


@gorilla.patch(
    torch._dynamo.variables.user_defined.UserDefinedObjectVariable,
)
def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
    from . import ConstantVariable

    source: Source | None = AttrSource(self.source, name) if self.source else None

    if self._object_has_getattribute:
        getattribute_fn = inspect.getattr_static(type(self.value), "__getattribute__")
        new_source: AttrSource | None = (
            AttrSource(self.source, "__getattribute__") if self.source else None
        )

        try:
            return variables.UserMethodVariable(
                getattribute_fn,
                self,
                # pyrefly: ignore[unbound-name]
                source=new_source,
            ).call_function(tx, [ConstantVariable.create(name)], {})
        except ObservedAttributeError:
            # Pass through to __getattr__ if __getattribute__ fails
            handle_observed_exception(tx)

    if tx.output.side_effects.has_pending_mutation_of_attr(self, name):
        result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
        if isinstance(result, variables.DeletedVariable):
            raise_observed_exception(
                AttributeError,
                tx,
                args=[
                    f"'{type(self.value).__name__}' object has no attribute '{name}'"
                ],
            )
        return result

    if name == "__dict__":
        options_dict = {"source": source}
        return variables.GetAttrVariable(self, name, None, **options_dict)

    # TODO(anijain2305) - Investigate if we need specialization for more
    # dunder attrs. inspect.getattr_static does not return correct value for
    # them.
    if name == "__class__":
        cls_source: Source | None = source
        if source is None:
            cls_source = self.cls_source
        else:
            cls_source = source
        options = {"source": cls_source}
        return UserDefinedClassVariable(type(self.value), **options)

    try:
        subobj = self._getattr_static(name)
        # Add by CAMBRICON
        if (
            hasattr(subobj, "is_mlu_gpu_migration")
            and subobj.is_mlu_gpu_migration is True
            and hasattr(subobj, "__wrapped__")
        ):
            subobj = subobj.__wrapped__
        # end Add by CAMBRICON
    except AttributeError:
        subobj = NO_SUCH_SUBOBJ
        getattr_fn = self._check_for_getattr()
        if isinstance(getattr_fn, types.FunctionType):
            # Dynamo is going to trace the __getattr__ function with
            # args=name. Set the source accordingly.
            if (
                getattr_fn is unpatched_nn_module_getattr
                and isinstance(self, variables.UnspecializedNNModuleVariable)
                # prevent against overwriting of params/buffers/submodules
                and istype(self.value._parameters, dict)  # type: ignore[attr-defined]
                and istype(self.value._buffers, dict)  # type: ignore[attr-defined]
                and istype(self.value._modules, dict)  # type: ignore[attr-defined]
            ):
                # Manually trace out the nn module __getattr__ to avoid large compilation latency.
                out = self.manually_trace_nn_module_getattr(tx, name)
            else:
                new_source = None
                if self.source:
                    new_source = AttrSource(self.source, "__getattr__")
                out = variables.UserMethodVariable(
                    getattr_fn, self, source=new_source
                ).call_function(tx, [ConstantVariable.create(name)], {})

            if self.source and getattr_fn is torch.nn.Module.__getattr__:
                if isinstance(
                    out,
                    (
                        variables.UnspecializedNNModuleVariable,
                        variables.NNModuleVariable,
                    ),
                ):
                    # nn_module_stack source is BC surface area. Ensure that
                    # mod._modules["linear"] is reflected as mod.linear for
                    # nn_module_stack.
                    out.set_nn_module_stack_source(  # type: ignore[attr-defined]
                        AttrSource(self.get_nn_module_stack_source(), name)  # type: ignore[attr-defined]
                    )
            return out

        elif getattr_fn is not None:
            unimplemented(
                gb_type="User-defined object with non-function __getattr__",
                context=f"object={self.value}, name={name}, getattr_fn={getattr_fn}",
                explanation=f"Found a non-function __getattr__ {getattr_fn} from a user-defined object {self.value} "
                f" when attempting to getattr `{name}`",
                hints=[
                    "Ensure the object's __getattr__ is a function type.",
                ],
            )

    from ..mutation_guard import unpatched_nn_module_init

    if subobj is torch.nn.Module.__init__:
        subobj = unpatched_nn_module_init

    # Check if its already saved, avoids inspect getattr_static call
    if name in self._subobj_from_class:
        subobj_from_class = self._subobj_from_class[name]
    else:
        subobj_from_class = inspect.getattr_static(
            self.value.__class__, name, NO_SUCH_SUBOBJ
        )
        self._subobj_from_class[name] = subobj_from_class

    is_accessible_from_type_mro = (
        subobj_from_class is subobj
        and self.cls_source is not None
        and self.source is not None
        and hasattr(self.value, "__dict__")
        and name not in self.value.__dict__
    )

    if isinstance(subobj, property):
        if self.source:
            # Read the class attribute to reach the property
            source = AttrSource(self.get_source_by_walking_mro(name), "fget")
        fget_vt = VariableTracker.build(tx, subobj.fget, source=source, realize=True)
        return fget_vt.call_function(tx, [self], {})
    elif isinstance(subobj, _collections._tuplegetter):
        # namedtuple fields are represented by _tuplegetter, and here we
        # emulate its `__get__`, which is implemented in C.
        _, (idx, _) = subobj.__reduce__()
        # Don't go through the `__getitem__` method anymore, see
        # https://github.com/python/cpython/blob/470941782f74288823b445120f6383914b659f23/Modules/_collectionsmodule.c#L2690
        assert isinstance(self, UserDefinedTupleVariable)
        return self._tuple_vt.items[idx]  # type: ignore[union-attr]
    elif isinstance(subobj, staticmethod):
        # Safe because `staticmethod.__get__` basically won't trigger user
        # code and just returns the underlying `__func__`:
        # https://github.com/python/cpython/blob/3.11/Objects/funcobject.c#L1088-L1100
        if is_accessible_from_type_mro:
            # Accessing from __dict__ does not resolve the descriptor, it
            # returns a staticmethod object, so access the __func__
            # attribute to get to the actual function.
            source = AttrSource(self.get_source_by_walking_mro(name), "__func__")
        func = subobj.__get__(self.value)
        return VariableTracker.build(tx, func, source)
    elif isinstance(subobj, classmethod):
        source_fn = None
        if is_accessible_from_type_mro:
            # Accessing from __dict__ does not resolve the descriptor, it
            # returns a classmethod object, so access the __func__
            # attribute to get to the actual function.
            source_fn = AttrSource(self.get_source_by_walking_mro(name), "__func__")  # type: ignore[assignment]
        return variables.UserMethodVariable(
            subobj.__func__,
            self.var_getattr(tx, "__class__"),
            source_fn=source_fn,
            source=source,
        )
    elif isinstance(subobj, types.ClassMethodDescriptorType):
        # e.g.: inspect.getattr_static({}, "fromkeys")
        func = subobj.__get__(self.value, None)
        return VariableTracker.build(tx, func, source)
    elif is_lru_cache_wrapped_function(subobj):
        # getattr_static returns the lru_wrapped function, and we cannot
        # extract the underlying method from the wrapped function. To handle
        # it, manually create a wrapped user method vt.
        return variables.WrapperUserMethodVariable(
            subobj, "__wrapped__", self, source=source
        )
    elif inspect.getattr_static(
        type(subobj), "__get__", NO_SUCH_SUBOBJ
    ) is not NO_SUCH_SUBOBJ and not is_wrapper_or_member_descriptor(
        type(subobj).__get__  # type: ignore[attr-defined]
    ):
        # Emulate https://github.com/python/cpython/blob/3.11/Objects/object.c#L1271-L1285
        #
        # Attribute has a __get__ method. Create a user defined object vt
        # for the subobj, and then trace the __get__ method.
        descriptor_source = None
        descriptor_get_source = None
        if self.cls_source:
            # To access the method descriptor from the udf object w/o using
            # inspect.getattr_static, we can look into the class mro
            descriptor_source = self.get_source_by_walking_mro(name)
            descriptor_get_source = AttrSource(TypeSource(descriptor_source), "__get__")
            descriptor_var = VariableTracker.build(tx, subobj, descriptor_source)
        else:
            # Sourceless Builder does not support user defined objects
            descriptor_var = UserDefinedObjectVariable(subobj)

        # The arguments of the __get__ function are (self, instance, owner)
        # self - descriptor_var
        # instance - instance of the class, represented by self here
        # owner - class object
        owner_var = UserDefinedClassVariable(type(self.value))
        return variables.UserMethodVariable(
            subobj.__get__.__func__,  # type: ignore[attr-defined]
            descriptor_var,
            source=descriptor_get_source,
        ).call_function(tx, [self, owner_var], {})
    elif isinstance(subobj, types.FunctionType) or (
        isinstance(subobj, types.MethodType) and isinstance(self.value, torch.nn.Module)
    ):
        # Since we get subobj via self._getattr_static, which may not trigger dynamic lookup.
        # Static lookup can't tell us it's a method or function correctly,
        # so we trigger dynamic lookup here to get the correct type.
        dynamic_subobj = getattr(self.value, name)

        while dynamic_subobj is subobj and hasattr(subobj, "_torchdynamo_inline"):
            subobj = subobj._torchdynamo_inline
            dynamic_subobj = subobj
            source = AttrSource(source, "_torchdynamo_inline") if source else None

        if isinstance(subobj, types.MethodType):
            if dynamic_subobj.__self__ is not self.value:
                if not isinstance(dynamic_subobj.__func__, types.FunctionType):
                    unimplemented(
                        gb_type="User-defined object method with non-function __func__",
                        context=f"object={self.value}, name={name}, method={dynamic_subobj}, "
                        f"method.__self__={dynamic_subobj.__self__}, method.__func__={dynamic_subobj.__func__}",
                        explanation=f"Method {dynamic_subobj} (name={name}) of user-defined object {self.value} has a "
                        f"__func__ ({dynamic_subobj.__func__}) that is not a function type.",
                        hints=[
                            "Ensure that the method's __func__ is a function type.",
                        ],
                    )

                # Use the __self__ attribute of the method to find the
                # source of the new self object.
                self_source = None
                if source is not None:
                    self_source = AttrSource(source, "__self__")
                object_vt = VariableTracker.build(
                    tx, dynamic_subobj.__self__, self_source
                )

                return variables.UserMethodVariable(
                    dynamic_subobj.__func__,
                    object_vt,
                )
            func = subobj.__func__
        else:
            assert isinstance(subobj, types.FunctionType)
            func = subobj

        if inspect.ismethod(dynamic_subobj):
            var_source = None
            if is_accessible_from_type_mro:
                var_source = self.get_source_by_walking_mro(name)
            return variables.UserMethodVariable(
                func, self, source_fn=var_source, source=source
            )
        elif inspect.isfunction(dynamic_subobj):
            return VariableTracker.build(tx, func, source)

    if (
        # wrap the source only if inline_inbuilt_nn_modules is set or fsdp modules. This is a temporary solution to
        # keep Dynamo behavior compatible with no inlining, as there will be some delay to turn on the flag in
        # fbcode.
        (
            torch._dynamo.config.inline_inbuilt_nn_modules
            or isinstance(self, variables.FSDPManagedNNModuleVariable)
        )
        and source
        and isinstance(self, variables.UnspecializedNNModuleVariable)
        # export has some awkwardness around specialized and unspecialized modules. Skip wrapping source for export
        # usecase for now.
        and (not tx.output.export or torch._dynamo.config.install_free_tensors)
    ):
        # Recalculate source for params/buffers
        if name in ("_buffers", "_parameters"):
            assert self.source is not None
            source = UnspecializedParamBufferSource(self.source, name)
        source = self._wrap_source(source)

    if subobj is not NO_SUCH_SUBOBJ:
        if (
            is_wrapper_or_member_descriptor(subobj)
            or torch._C._dynamo.utils.is_instancemethod(subobj)  # type: ignore[attr-defined]
            or is_cython_function(subobj)
        ):
            options = {"source": source}
            return variables.GetAttrVariable(self, name, None, **options)
        if source:
            if is_accessible_from_type_mro:
                source = self.get_source_by_walking_mro(name)

            return variables.LazyVariableTracker.create(subobj, source)
        else:
            # Check if the subobj is accessible from the class itself. If the class source is known, we can create a
            # sourceful variable tracker.
            if self.cls_source is not None:
                subobj_from_class = inspect.getattr_static(
                    self.value.__class__, name, NO_SUCH_SUBOBJ
                )
                if subobj_from_class is subobj:
                    src_from_class = AttrSource(self.cls_source, name)
                    return variables.LazyVariableTracker.create(
                        subobj_from_class, src_from_class
                    )

            return VariableTracker.build(tx, subobj)

    # Earlier we were returning GetAttrVariable but its incorrect. In absence of attr, Python raises AttributeError.
    raise_observed_exception(
        AttributeError,
        tx,
        args=[f"'{type(self.value).__name__}' object has no attribute '{name}'"],
    )
