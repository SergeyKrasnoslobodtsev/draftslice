class DraftsliceException(Exception):
    """Базовое исключение для библиотеки Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DraftsliceValueError(DraftsliceException):
    """Исключение для ошибок значения в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DraftsliceTypeError(DraftsliceException):
    """Исключение для ошибок типа в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DraftsliceRuntimeError(DraftsliceException):
    """Исключение для ошибок времени выполнения в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ArrayNullError(DraftsliceException):
    """Исключение для ошибок работы с пустыми массивами в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class DraftsliceFileNotFoundError(DraftsliceException):
    """Исключение для ошибок отсутствия файла в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class IORuntimeError(DraftsliceException):
    """Исключение для ошибок ввода-вывода в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidImageAngleError(DraftsliceException):
    """Исключение для ошибок некорректного угла поворота изображения в библиотеке Draftslice."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
