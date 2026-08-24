import sys
import inspect
import types
import torch
from functools import wraps
from torch_mlu._MLUC import _MLURecordFunction


def _set_outputs(rf, result):
    if isinstance(result, torch.Tensor):
        rf._set_outputs([result])
    elif isinstance(result, (list, tuple)):
        tensor_outputs = [r for r in result if isinstance(r, torch.Tensor)]
        if tensor_outputs:
            rf._set_outputs(tensor_outputs)


class FunctionRegistry:
    def __init__(self):
        self._pending = []
        self._wrapped = {}

    def _make_wrapper(self, name, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if "." in func.__qualname__ and not isinstance(
                func, types.BuiltinFunctionType
            ):
                record_args = args[1:]
            else:
                record_args = args
            with _MLURecordFunction(
                name,
                input_values=list(record_args),
                keyword_values=kwargs,
            ) as rf:
                result = func(*args, **kwargs)
                _set_outputs(rf, result)
                return result

        return wrapper

    def _patch_imported_refs(self, orig_func, new_func):
        for module in sys.modules.values():
            if module and hasattr(module, "__dict__"):
                for k, v in list(module.__dict__.items()):
                    if v is orig_func:
                        module.__dict__[k] = new_func

    def _is_function(self, func):
        return inspect.isfunction(func) or isinstance(func, types.BuiltinFunctionType)

    def register_op(self, func, name=None):
        if not self._is_function(func):
            raise TypeError(
                f"register_custom_op() expects a function parameter, but got {type(func).__name__}."
            )
        func_name = func.__name__
        module = inspect.getmodule(func)
        if "." in func.__qualname__ and not isinstance(func, types.BuiltinFunctionType):
            module_name = func.__qualname__.split(".")[0]
            module = getattr(module, module_name, module)
            full_name = f"{module_name}.{func_name}"
        else:
            full_name = func_name
        if name:
            full_name = name
        if module is None:
            raise RuntimeError(
                f"Module '{func.__module__}' is not registered in sys.modules.\n"
                f"Try adding the following line after building your extension:\n\n"
                f"    import sys\n"
                f"    sys.modules['{func.__module__}'] = your_module_object\n\n"
                f"Or, alternatively, build and install it as a proper Python package "
                f"so that it is imported via the standard import mechanism."
            )
        self._pending.append((full_name, module, func_name, func))

    def register_module(self, module):
        if isinstance(module, types.ModuleType):
            module_name = module.__name__
        elif inspect.isclass(module):
            module_name = module.__module__
        else:
            raise TypeError(
                f"register_custom_module() expects a python module or class parameter, but got {type(module).__name__}."
            )
        for attr_name, attr_value in inspect.getmembers(module):
            if getattr(attr_value, "__module__", "__attr_module_name__") == module_name:
                if self._is_function(attr_value):
                    self.register_op(attr_value)
                elif inspect.isclass(attr_value):
                    self.register_module(attr_value)

    def wrap(self):
        for full_name, module, attr_name, orig_func in set(self._pending):
            wrapper_key = f"{module.__name__}_{full_name}"
            if wrapper_key not in self._wrapped:
                wrapper = self._make_wrapper(full_name, orig_func)
                setattr(module, attr_name, wrapper)
                self._wrapped[wrapper_key] = (module, attr_name, orig_func)
                self._patch_imported_refs(orig_func, wrapper)

    def unwrap(self):
        for wrapper_key, (module, attr_name, orig_func) in self._wrapped.items():
            current = getattr(module, attr_name)
            setattr(module, attr_name, orig_func)
            self._patch_imported_refs(current, orig_func)
        self._wrapped.clear()


registry = FunctionRegistry()
