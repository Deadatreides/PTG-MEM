"""Запуск программы codebase-memory-mcp: командная строка и сам запуск процесса с выводом в UTF-8.

node: leaf
effects: io
"""
import subprocess  # noqa: S404

import deal


@deal.ensure(lambda binary, tool, args, result: result == (binary, 'cli', '--quiet', tool, args))
@deal.has()
def command(binary: str, tool: str, args: str) -> tuple[str, ...]:
    """ФТ-006. Командная строка запуска: (binary, 'cli', '--quiet', tool, args).

    D: 1

    >>> command('cbm.exe', 'list_projects', '{}')
    ('cbm.exe', 'cli', '--quiet', 'list_projects', '{}')
    >>> command('C:/tools/cbm', 'index_repository', '{"repo_path": "."}')
    ('C:/tools/cbm', 'cli', '--quiet', 'index_repository', '{"repo_path": "."}')
    """
    return (binary, 'cli', '--quiet', tool, args)


@deal.pre(lambda argv: len(argv) >= 1)
@deal.ensure(lambda argv, result: isinstance(result[0], int) and isinstance(result[1], str) and isinstance(result[2], str))
@deal.raises(FileNotFoundError, RuntimeError)
@deal.has('io', 'syscall')
def execute(argv: tuple[str, ...]) -> tuple[int, str, str]:
    """ФТ-006. Запуск процесса argv и ожидание его конца: (код выхода, stdout, stderr), оба потока — текст UTF-8
    (ошибки декодирования заменяются). Программы argv[0] нет — FileNotFoundError(argv[0]); не запускается по другой
    причине (OSError, ValueError — например, нулевой символ в argv) — RuntimeError с текстом ошибки. Запуск —
    subprocess.run(argv, capture_output=True, check=False); предупреждение ruff S603 снимается комментарием noqa.

    D: 3

    >>> execute(('нет-такой-программы.exe', 'cli', '--quiet', 'list_projects', '{}'))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    FileNotFoundError: нет-такой-программы.exe
    >>> execute(('./no-such-tool',))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    FileNotFoundError: ./no-such-tool
    """
    try:
        done = subprocess.run(argv, capture_output=True, check=False)  # noqa: S603
    except FileNotFoundError:
        raise FileNotFoundError(argv[0]) from None
    except (OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    stdout = done.stdout.decode('utf-8', errors='replace')
    stderr = done.stderr.decode('utf-8', errors='replace')
    return (done.returncode, stdout, stderr)
