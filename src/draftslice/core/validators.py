import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any

import numpy as np


def validate_image(arg_name: str | None = None, channels: int | None = None, dtype: type | None = None) -> Callable:
    """Декоратор для валидации изображений по количеству каналов и типу данных."""

    def decorator(func: Callable) -> Callable:
        resolved_arg_name = arg_name or next(iter(inspect.signature(func).parameters))

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Привязываем аргументы к параметрам функции
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if resolved_arg_name in bound_args.arguments:
                img = bound_args.arguments[resolved_arg_name]
                if isinstance(img, np.ndarray):
                    # Проверка каналов
                    actual_channels = img.shape[2] if img.ndim == 3 else 1
                    if channels is not None and actual_channels != channels:
                        raise TypeError(
                            f"[{func.__name__}] Поле '{resolved_arg_name}' ожидает {channels} "
                            f"канал(ов), получено: {actual_channels}"
                        )
                    # Проверка dtype
                    if dtype is not None and img.dtype != dtype:
                        raise TypeError(
                            f"[{func.__name__}] Поле '{resolved_arg_name}' ожидает dtype={dtype}, получено: {img.dtype}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator
