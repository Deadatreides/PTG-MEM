"""
ptg_logging.py — подробное файловое логирование (ПАТЧ 19).

Отдельно от Archive.log()/progress_cb: тот — куратированный, user-facing
поток для GUI/консоли (только важные вехи: "Построение файлового
графа..."). Логгер отсюда — гораздо детальнее и пишется ТОЛЬКО в файл,
не в GUI (иначе прогресс-лог захлебнётся): каждая проба embedding-модели
и её результат, каждое решение placement_case на каждом атоме, каждая
проглоченная исключением ошибка экстрактора (.docx/.doc/.py), полный
traceback при сбоях.

Зачем это нужно именно сейчас: при разборе инцидента с молчаливой сменой
embedding-модели (ПАТЧ 18) пришлось читать СЫРОЙ лог самого LM Studio,
потому что PTG о своих собственных решениях (какую модель попробовал,
почему отбросил, что выбрал) нигде не писал. Теперь пишет — в
<ptg_dir>/ptg.log, тем же файлом, что бы ни открывало Archive (GUI,
MCP-сервер).
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_ROOT_LOGGER_NAME = "ptg"


def setup_logging(ptg_dir: str, level: int = logging.DEBUG) -> logging.Logger:
    """(Пере)настроить единый файловый логгер 'ptg' на запись в
    <ptg_dir>/ptg.log. Идемпотентно и безопасно вызывать многократно (это
    происходит при каждом Archive.__init__ — GUI создаёт новый Archive при
    каждом выборе папки/смене output_dir): старый RotatingFileHandler (если
    указывал на другую папку) снимается, новый добавляется. Дочерние
    логгеры (get_logger("core"), get_logger("embedder"), ...) наследуют
    этот handler автоматически через стандартный logging propagate —
    ничего дополнительно связывать не нужно.

    Ротация: до 5 МБ на файл, 3 резервные копии (ptg.log.1, .2, .3) — лог
    не растёт бесконечно на долгоживущем архиве."""
    os.makedirs(ptg_dir, exist_ok=True)
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # не дублировать в root-логгер/консоль по умолчанию

    log_path = os.path.join(ptg_dir, "ptg.log")
    for h in list(logger.handlers):
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
            h.close()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """logging.getLogger(f"ptg.{module_name}") — шорткат для дочерних
    логгеров модулей (core/embedder/py_comments/snapshot_store/mcp/gui).
    Работает даже ДО вызова setup_logging() (просто не пишет никуда, пока
    родительский логгер не настроен обработчиком) — безопасно импортировать
    и создавать логгер на уровне модуля."""
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{module_name}")
