"""
ptg_chat_exports.py — извлечение Q+A пар из СТРУКТУРИРОВАННЫХ JSON-экспортов
чатов с ИИ: ChatGPT/OpenAI, Claude/Anthropic, Grok/xAI, Gemini (Google
Takeout), Meta AI. ПАТЧ 23.

ЗАЧЕМ ЭТОТ МОДУЛЬ. До ПАТЧ 23 .json/.jsonl входили в TEXT_EXTS, но
обрабатывались как plain text: _read_text_file() читал файл целиком, а
_split_into_blocks()/_label_blocks() резали его по пустым строкам и искали
речевые метки вроде "User:"/"ChatGPT said:". Для официальных экспортов это
даёт мусор: ChatGPT хранит переписку как ДЕРЕВО узлов (mapping/parent/
children) внутри одной строки JSON без единого переноса строки между
репликами — эвристика по пустым строкам такой файл вообще не делит; Claude
хранит список сообщений с полями sender/text, но тоже как машинный JSON, а
не текст с метками. В обоих случаях старый парсер либо не находил ни одной
пары (весь файл — один "атом"-мусор), либо резал JSON синтаксически, не
семантически.

Вынесено в отдельный модуль (по тому же принципу, что и ptg_py_comments.py):
изолирует специфику форматов от общего текстового пайплайна ptg_core.py,
чистые функции без побочных эффектов, тестируемые на синтетических JSON-
буферах без файловой системы.

ЧЕСТНЫЙ ПРИНЦИП РАЗДЕЛЕНИЯ ФОРМАТОВ (важно, не переусложнять там, где
уверенности нет):

1. Форматы с ТОЧНО задокументированной структурой (ChatGPT — дерево
   mapping/parent/children/current_node; Claude — список chat_messages с
   sender/text) разбираются ИМЕННО по этой структуре — extract_chat_pairs().

2. Форматы, где роль/контент выражены через одну из нескольких общепринятых
   пар ключей (role/sender/author/from/speaker + content/text/message/body) —
   это покрывает Grok (xAI: "role", "content", timestamp — см. описания
   официального экспорта) и Meta-стиль (Messenger/Instagram/WhatsApp-экспорт,
   которым переиспользуется история Meta AI: "sender_name" + "content" +
   "timestamp_ms") — разбираются ОБЩИМ структурным детектором, а не жёстко
   прошитой схемой конкретного вендора: экспортные форматы меняются, а
   структурный сигнал (пары ключей роль+контент в списке словарей) остаётся
   устойчивым дольше, чем точные имена полей одного конкретного релиза.

3. Google Takeout ("My Activity" → "Gemini Apps" → MyActivity.json) — это
   ЖУРНАЛ АКТИВНОСТИ (по записи на событие: header/title/time/products), а
   НЕ дерево диалога и не список сообщений с явной ролью каждой записи.
   Разбить его на строгие пары (question, answer) означало бы гадать о
   разделении, которое нельзя проверить без реального экспорта под рукой —
   вместо этого extract_chat_text_stream() честно вытаскивает текст каждой
   записи в хронологическом порядке единым текстовым потоком и отдаёт его
   вызывающему коду (parse_file_to_atoms в ptg_core.py), который прогоняет
   поток через ТОТ ЖЕ эвристический Q+A парсер (метки/sequential-склейка),
   что уже используется для обычных текстовых экспортов. Это медленнее
   гарантированной пары, но не даёт ложной уверенности там, где её нет —
   тот же принцип, что и в §5 PTG_EMBEDDER.md про честное документирование
   ограничений вместо изобретения несуществующих меток времени.

4. Если НИ ОДИН детектор не сработал (обычный .json-конфиг проекта,
   package.json, произвольные структурированные данные) — обе функции
   возвращают None, и ptg_core.py откатывается на прежнее поведение
   (_read_text_file + текстовый эвристический парсер) без изменений —
   полная обратная совместимость для нечатовых .json-файлов.
"""

import re
import datetime as _dt

from ptg_logging import get_logger

_log = get_logger("chat_exports")

# ---------------------------------------------------------------------------
# Общие ключи, по которым распознаём "сообщение" в generic-формате
# ---------------------------------------------------------------------------
_ROLE_KEYS = ("role", "sender", "author", "from", "speaker", "sender_name")
_CONTENT_KEYS = ("content", "text", "message", "body", "parts")
_NESTED_MESSAGE_KEYS = ("messages", "chat_messages", "turns", "conversation", "history", "chats")

_USER_ROLE_VALUES = {"user", "human", "you", "me"}
_SKIP_ROLE_VALUES = {"system", "developer", "tool", "function"}
# ПАТЧ 23 — Meta AI переиспользует схему Messenger/Instagram-экспорта
# ("sender_name" вместо "role"): в диалоге с Meta AI один из участников —
# сам бот, отображаемый под одним из этих имён. Всё остальное значение
# sender_name считается человеком (user) — это НЕ белый список людей
# (невозможно перечислить все имена пользователей), а чёрный список
# известных отображаемых имён ИИ в Meta-экспортах.
_META_AI_DISPLAY_NAMES = {"meta ai", "meta al", "llama", "ai assistant"}


def _has_message_shape(d):
    return isinstance(d, dict) and any(k in d for k in _ROLE_KEYS) and any(k in d for k in _CONTENT_KEYS)


def _fix_meta_mojibake(text):
    """Известный баг экспортёра Meta ("Скачать вашу информацию" /
    "Download your information"): не-ASCII символы (кириллица, эмодзи, ё)
    в JSON-строках выглядят как "кракозябра", потому что исходные UTF-8
    байты были ошибочно повторно интерпретированы как latin-1 при
    сериализации. Обратное преобразование chars.encode('latin-1')
    .decode('utf-8') чинит это надёжно и БЕЗОПАСНО для всех остальных
    случаев: если текст уже нормальный (не из Meta, либо чистый ASCII),
    в нём почти наверняка есть символы за пределами диапазона latin-1
    (0x00-0xFF) — например, кириллица (0x0400+) — и .encode('latin-1')
    в этом случае бросит UnicodeEncodeError; тогда просто возвращаем текст
    без изменений. Настоящая "кракозябра" Meta, наоборот, ВСЯ состоит из
    символов, попавших в latin-1 при повторном декодировании, поэтому
    round-trip проходит и восстанавливает исходный текст."""
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return fixed or text


def _coerce_timestamp(value, is_ms=False):
    """Привести значение временной метки (unix-секунды, unix-миллисекунды
    или ISO8601-строка) к unix-секундам (float). None, если формат не
    распознан — вызывающий код в этом случае честно откатывается на mtime
    файла, а не выдумывает время."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if is_ms or v > 1e12:
            return v / 1000.0
        return v
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        iso = s[:-1] + "+00:00" if s.endswith("Z") else s
        try:
            return _dt.datetime.fromisoformat(iso).timestamp()
        except ValueError:
            pass
        try:
            return _coerce_timestamp(float(s))
        except ValueError:
            return None
    return None


def _extract_generic_text(value):
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                t = item.get("text") or item.get("value")
                itype = item.get("type", "text")
                if isinstance(t, str) and t.strip() and itype in ("text", "text_delta", None):
                    parts.append(t.strip())
        return "\n\n".join(parts) or None
    if isinstance(value, dict):
        return _extract_generic_text(value.get("text") or value.get("parts") or value.get("value"))
    return None


def _normalize_message(d):
    """Свести произвольный словарь-сообщение к (role, text, timestamp),
    role in {'user', 'assistant'}. None, если это не похоже на сообщение
    (нет ни роли, ни контента) или роль явно служебная (system/tool)."""
    if not isinstance(d, dict):
        return None
    role_key = next((k for k in _ROLE_KEYS if k in d), None)
    if role_key is None:
        return None
    raw_role = d[role_key]
    if isinstance(raw_role, dict):
        raw_role = raw_role.get("role") or raw_role.get("name") or raw_role.get("label")
    role_str = str(raw_role or "").strip().lower()

    is_meta_shape = role_key == "sender_name"
    if is_meta_shape:
        if not role_str:
            return None
        role = "assistant" if role_str in _META_AI_DISPLAY_NAMES else "user"
    elif role_str in _USER_ROLE_VALUES:
        role = "user"
    elif role_str in _SKIP_ROLE_VALUES or not role_str:
        return None
    else:
        role = "assistant"  # ловит "assistant"/"ai"/"bot"/"grok"/"gemini"/имя конкретной модели и т.п.

    content_key = next((k for k in _CONTENT_KEYS if k in d), None)
    text = _extract_generic_text(d.get(content_key)) if content_key else None
    if is_meta_shape and text:
        text = _fix_meta_mojibake(text)
    if not text:
        return None

    ts_key = next((k for k in ("timestamp_ms", "timestamp", "created_at", "create_time", "time", "date")
                    if k in d), None)
    ts = _coerce_timestamp(d.get(ts_key), is_ms=(ts_key == "timestamp_ms")) if ts_key else None
    return (role, text, ts)


def _pair_role_messages(messages):
    """messages: список (role, text, timestamp) в хронологическом порядке.
    Та же логика группировки, что и в ptg_core._group_by_labels: подряд
    идущие 'user'-сообщения — вопрос, следующие 'assistant' — ответ; смена
    q->a->q закрывает предыдущую пару. Несколько подряд идущих сообщений
    одной роли (например, два user-сообщения без ответа между ними, частый
    случай при повторных попытках/правках) склеиваются в один вопрос — не
    теряются, но и не создают пустых промежуточных атомов."""
    pairs = []
    cur_q, cur_a, cur_ts = [], [], []

    def _flush():
        if cur_q or cur_a:
            pairs.append({
                "question": "\n\n".join(cur_q).strip(),
                "answer": "\n\n".join(cur_a).strip(),
                "timestamp": cur_ts[0] if cur_ts else None,
            })

    for role, text, ts in messages:
        if role == "user":
            if cur_q and cur_a:
                _flush()
                cur_q, cur_a, cur_ts = [], [], []
            cur_q.append(text)
        else:
            cur_a.append(text)
        if ts is not None:
            cur_ts.append(ts)
    _flush()
    return [p for p in pairs if p["question"] or p["answer"]]


def _as_conversation_list(data, required_key):
    """Нормализовать вход к списку словарей-диалогов: один диалог без
    внешнего списка (data сам содержит required_key) либо список диалогов
    (берём только элементы, реально содержащие required_key — экспорт может
    содержать посторонние записи)."""
    if isinstance(data, dict) and required_key in data:
        return [data]
    if isinstance(data, list):
        items = [d for d in data if isinstance(d, dict) and required_key in d]
        return items or None
    return None


# ---------------------------------------------------------------------------
# 1. ChatGPT / OpenAI — дерево mapping{id: {message, parent, children}}
# ---------------------------------------------------------------------------
def _extract_openai_content_text(content):
    if not isinstance(content, dict):
        return None
    if content.get("content_type") == "code":
        text = content.get("text")
        return text.strip() if isinstance(text, str) and text.strip() else None
    parts = content.get("parts")
    if isinstance(parts, list):
        texts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
        return "\n\n".join(texts) or None
    return None


def _try_openai_chatgpt(data):
    conversations = _as_conversation_list(data, required_key="mapping")
    if not conversations:
        return None

    all_pairs = []
    for conv in conversations:
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            continue

        collected = []
        for idx, node in enumerate(mapping.values()):
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            author = msg.get("author") or {}
            role_raw = (author.get("role") or "").lower()
            if role_raw == "user":
                role = "user"
            elif role_raw == "assistant":
                role = "assistant"
            else:
                continue  # system/tool/function — не входит в Q+A
            meta = msg.get("metadata") or {}
            if meta.get("is_visually_hidden_from_conversation"):
                continue  # скрытый служебный узел (например, память/системный контекст)
            text = _extract_openai_content_text(msg.get("content"))
            if not text:
                continue
            create_time = msg.get("create_time")
            ts = float(create_time) if isinstance(create_time, (int, float)) else None
            collected.append((role, text, ts, idx))

        if not collected:
            continue
        # Сортировка по create_time, если она есть у ВСЕХ узлов ветки;
        # иначе — порядок вставки в mapping (dict сохраняет insertion order,
        # обычно совпадает с порядком экспорта OpenAI).
        if all(c[2] is not None for c in collected):
            collected.sort(key=lambda c: c[2])
        else:
            collected.sort(key=lambda c: c[3])

        pairs = _pair_role_messages([(r, t, ts) for r, t, ts, _ in collected])
        title = conv.get("title")
        if title and pairs and pairs[0]["question"]:
            pairs[0]["question"] = f"[{title}]\n{pairs[0]['question']}"
        all_pairs.extend(pairs)

    return all_pairs or None


# ---------------------------------------------------------------------------
# 2. Claude / Anthropic — список chat_messages{sender, text|content}
# ---------------------------------------------------------------------------
def _extract_claude_content_text(msg):
    text = msg.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = msg.get("content")
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        if texts:
            return "\n\n".join(texts)
    return None


def _try_claude(data):
    conversations = _as_conversation_list(data, required_key="chat_messages")
    if not conversations:
        return None

    all_pairs = []
    for conv in conversations:
        raw_messages = conv.get("chat_messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            continue

        messages = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            sender = str(msg.get("sender") or "").lower()
            if sender == "human":
                role = "user"
            elif sender == "assistant":
                role = "assistant"
            else:
                continue
            text = _extract_claude_content_text(msg)
            if not text:
                continue
            ts = _coerce_timestamp(msg.get("created_at"))
            messages.append((role, text, ts))

        if not messages:
            continue
        pairs = _pair_role_messages(messages)
        name = conv.get("name")
        if name and name != "Untitled" and pairs and pairs[0]["question"]:
            pairs[0]["question"] = f"[{name}]\n{pairs[0]['question']}"
        all_pairs.extend(pairs)

    return all_pairs or None


# ---------------------------------------------------------------------------
# 3. Generic role+content список сообщений — Grok/xAI, Meta (Messenger-стиль
#    экспорта, которым переиспользуется история Meta AI), и любой другой
#    формат с парой ключей роль/контент, который не подошёл под 1 и 2.
# ---------------------------------------------------------------------------
def _looks_like_message_list(lst):
    if not isinstance(lst, list) or len(lst) < 2:
        return False
    dicts = [x for x in lst if isinstance(x, dict)]
    if len(dicts) < len(lst) * 0.8:
        return False
    sample = dicts[:10]
    matches = sum(1 for d in sample if _has_message_shape(d))
    return matches >= max(2, int(len(sample) * 0.6))


def _find_message_lists(data):
    """Список найденных списков сообщений (может быть несколько диалогов
    в одном файле). Ищем на верхнем уровне и на один уровень вглубь под
    известными ключами-обёртками — намеренно НЕ полная рекурсия по всему
    дереву, иначе произвольный JSON-конфиг с любым списком объектов внутри
    ложно распознавался бы как чат."""
    found = []

    def _scan_wrapper(conv):
        if not isinstance(conv, dict):
            return
        for key in _NESTED_MESSAGE_KEYS:
            v = conv.get(key)
            if _looks_like_message_list(v):
                found.append(v)
                return

    if _looks_like_message_list(data):
        found.append(data)
    elif isinstance(data, list):
        for item in data:
            _scan_wrapper(item)
    elif isinstance(data, dict):
        _scan_wrapper(data)

    return found


def _try_generic_message_list(data):
    lists = _find_message_lists(data)
    if not lists:
        return None

    all_pairs = []
    for msg_list in lists:
        messages = [n for n in (_normalize_message(d) for d in msg_list) if n]
        if messages:
            all_pairs.extend(_pair_role_messages(messages))
    return all_pairs or None


# ---------------------------------------------------------------------------
# 4. Google Takeout — "My Activity" → "Gemini Apps" → MyActivity.json.
#    Журнал активности, не дерево диалога — см. docstring модуля, п.3.
#    Возвращает ПЛОСКИЙ ТЕКСТ (не пары), см. extract_chat_text_stream().
# ---------------------------------------------------------------------------
_TAKEOUT_TITLE_PREFIX_RE = re.compile(
    r"^(Asked|Prompted|Chatted with)\s+Gemini\b[:]?\s*(with)?\s*",
    re.IGNORECASE,
)


def _try_google_takeout_activity(data):
    if not isinstance(data, list) or not data:
        return None
    sample = [x for x in data[:20] if isinstance(x, dict)]
    if not sample:
        return None

    def _is_gemini(x):
        products = x.get("products")
        if isinstance(products, list) and any("gemini" in str(p).lower() for p in products):
            return True
        return "gemini" in str(x.get("header") or "").lower()

    hits = sum(1 for x in sample if "time" in x and "title" in x and _is_gemini(x))
    if hits < max(1, int(len(sample) * 0.5)):
        return None

    dated = []
    for x in data:
        if not isinstance(x, dict) or "time" not in x or "title" not in x:
            continue
        ts = _coerce_timestamp(x.get("time"))
        dated.append((ts if ts is not None else 0.0, x))
    dated.sort(key=lambda p: p[0])

    lines = []
    for _, x in dated:
        title = _TAKEOUT_TITLE_PREFIX_RE.sub("", str(x.get("title") or "")).strip()
        if title:
            lines.append(title)
        sub = x.get("subtitles")
        if isinstance(sub, list):
            for s in sub:
                if isinstance(s, dict) and s.get("name"):
                    lines.append(str(s["name"]).strip())
                elif isinstance(s, str) and s.strip():
                    lines.append(s.strip())
        elif isinstance(sub, str) and sub.strip():
            lines.append(sub.strip())

    text = "\n\n".join(l for l in lines if l)
    return text or None


# ---------------------------------------------------------------------------
# Публичный вход
# ---------------------------------------------------------------------------
_PAIR_PARSERS = (
    ("chatgpt_mapping", _try_openai_chatgpt),
    ("claude_chat_messages", _try_claude),
    ("generic_role_content", _try_generic_message_list),
)

_TEXT_STREAM_PARSERS = (
    ("google_takeout_gemini_activity", _try_google_takeout_activity),
)


def extract_chat_pairs(data, source_label=""):
    """data — уже распарсенный json.loads()/построчный результат (dict или
    list). Возвращает список пар {"question", "answer", "timestamp"}
    (timestamp — unix-секунды float или None) в хронологическом порядке,
    либо None, если ни один известный/generic формат структурированной пары
    роль+контент не распознан — тогда parse_file_to_atoms() в ptg_core.py
    пробует extract_chat_text_stream(), а если и она вернёт None — честно
    откатывается на прежний текстовый эвристический парсер."""
    for name, fn in _PAIR_PARSERS:
        try:
            pairs = fn(data)
        except Exception as ex:
            _log.debug(f"extract_chat_pairs: «{name}» упал на {source_label}: {ex}")
            continue
        if pairs:
            _log.info(f"extract_chat_pairs: формат «{name}» распознан ({source_label}), "
                       f"{len(pairs)} Q+A пар")
            return pairs
    return None


def extract_chat_text_stream(data, source_label=""):
    """Для форматов, где строгая пара question/answer не может быть честно
    установлена по структуре (журналы активности вроде Google Takeout) —
    возвращает единый текстовый поток в хронологическом порядке, который
    дальше проходит через существующий текстовый Q+A эвристический парсер
    (тот же путь, что для .txt/.md). None, если ни один такой формат не
    распознан."""
    for name, fn in _TEXT_STREAM_PARSERS:
        try:
            text = fn(data)
        except Exception as ex:
            _log.debug(f"extract_chat_text_stream: «{name}» упал на {source_label}: {ex}")
            continue
        if text:
            _log.info(f"extract_chat_text_stream: формат «{name}» распознан ({source_label}), "
                       f"{len(text)} симв.")
            return text
    return None
