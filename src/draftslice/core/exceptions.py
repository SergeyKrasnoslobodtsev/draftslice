class DraftsliceException(Exception):
    """Базовое исключение для библиотеки Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ValueError(DraftsliceException):
    """Исключение для ошибок значения в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TypeError(DraftsliceException):
    """Исключение для ошибок типа в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RuntimeError(DraftsliceException):
    """Исключение для ошибок времени выполнения в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
