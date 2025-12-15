# Руководство по работе с VeriReason

Это руководство описывает установку, настройку и эксплуатацию инструментария VeriReason. В документе перечислены требования, подготовка окружения, этапы работы с датасетами, стандартный конвейер «рассуждение + тестбенч», а также воспроизводимая демонстрация на `sample_15.jsonl`.

## 1. Что такое VeriReason?
- **Цель:** повысить качество генерации Verilog RTL за счет совмещения супервизорного дообучения (SFT), обучения с подкреплением (GRPO) и явного рассуждения.
- **Ключевые компоненты:**
  - `gen_reasoning_tb.py`: получает рассуждение, проверяет или исправляет RTL, создает тестбенч и прогоняет iverilog/vvp для подтверждения корректности.
  - Каталоги `GRPO/` и `SFT/`: инструкции по обучению моделей (OpenR1, LLAMAFactory и др.).
  - `datasets/`: локальные наборы данных (например, `RTL-Coder_small`), склонированные из HuggingFace.

## 2. Требования
| Компонент | Назначение | Установка |
|-----------|-----------|-----------|
| Python ≥ 3.9 | Запуск скриптов | Системный пакет, pyenv и т.п. |
| `pip install -r requirements` (пример) | Python-зависимости | Необходимы как минимум `openai`, `tqdm` |
| Icarus Verilog (`iverilog`, `vvp`) | Компиляция/симуляция | `brew install icarus-verilog` (macOS) или пакетный менеджер |
| LM Studio или другой OpenAI-совместимый endpoint | LLM-бэкенд | В примере используется LM Studio с моделью `openai/gpt-oss-20b` |
| Git, wget | Клонирование репозитория и датасетов | Обычно уже установлены |
| (Опционально) CUDA, PyTorch, Accelerate, NCCL | Для SFT/GRPO-обучения | См. README |

### Создание Python-окружения
```bash
git clone https://github.com/NellyW8/VeriReason.git
cd VeriReason
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # при отсутствии файла установка openai, tqdm вручную
```

## 3. Настройка LLM (пример LM Studio)
1. Запустите LM Studio с моделью `openai/gpt-oss-20b` (или любой, поддерживающей Chat Completions API).
2. Включите OpenAI-совместимый сервер (по умолчанию `http://127.0.0.1:1234/v1`).
3. Экспортируйте переменные окружения, чтобы VeriReason обращался к вашему серверу:
   ```bash
   export VERIREASON_API_BASE="http://127.0.0.1:1234/v1"
   export VERIREASON_MODEL_ID="openai/gpt-oss-20b"
   export VERIREASON_API_KEY="lm-studio"  # ключ фиктивный; LM Studio его игнорирует, но клиент требует
   ```
   > Для OpenAI или других провайдеров укажите реальные адреса и ключи.

## 4. Подготовка датасетов
VeriReason работает с JSON/JSONL-файлами, в которых есть поля `instruction` и `output`. Рекомендуемые источники:
- **HuggingFace:** `Nellyw888/RTL-Coder_small` (уже склонирован в `datasets/RTL-Coder_small`).
- **Локальный мини-набор:** `datasets/RTL-Coder_small/sample_15.jsonl` содержит 15 записей для тестовых прогонов.

Пример команды скачивания:
```bash
git clone https://huggingface.co/datasets/Nellyw888/RTL-Coder_small datasets/RTL-Coder_small
```

## 5. Запуск конвейера «рассуждение + тестбенч»
Главный рабочий процесс выполняет `gen_reasoning_tb.py`:
1. Загружает JSON/JSONL-датасет.
2. Для каждого задания:
   - Проверяет или регенерирует Verilog-код.
   - Получает текстовое объяснение рассуждений.
   - Генерирует тестбенч с помощью LLM.
   - Компилирует и симулирует код и тестбенч (`iverilog`/`vvp`).
3. Сохраняет результаты (код + тестбенч + тест-векторы) в `OUTPUT_DIR/processed_entries.jsonl`.

### Переменные окружения
```bash
export VERIREASON_INPUT_FILE="$(pwd)/datasets/RTL-Coder_small/sample_15.jsonl"
export VERIREASON_OUTPUT_DIR="$(pwd)/runs/rtl_small"
# необязательный лог
# export VERIREASON_LOG_FILE=$VERIREASON_OUTPUT_DIR/run.log
```

### Запуск
```bash
python gen_reasoning_tb.py
```

Скрипт выводит прогресс и пишет лог в `runs/rtl_small/run.log`. В логе видно, когда тестбенч пересоздается, а когда симуляция проходит успешно/с ошибками.

## 6. Демонстрация на `sample_15.jsonl`
Выполните шаги выше, затем изучите результаты.

### Основные файлы
| Файл | Содержимое |
|------|------------|
| `runs/rtl_small/run.log` | Хронологический лог (HTTP-запросы, прогресс, итог: напр. 15 обработано / 10 симуляций успешны) |
| `runs/rtl_small/processed_entries.jsonl` | Обогащенный набор (код + reasoning + тестбенч + тест-векторы для каждого entry) |
| `runs/rtl_small/analysis.md` | Дополнительный текстовый отчет |

### Быстрые проверки
```bash
# Подсчет строк
wc -l runs/rtl_small/processed_entries.jsonl

# Просмотр конкретной записи (entry_4)
rg -n '"id": "entry_4"' -n '"tb_result"' runs/rtl_small/processed_entries.jsonl

# Проверка наличия тестбенчей у последних 15 записей
python - <<'PY'
import json
rows=[json.loads(line) for line in open("runs/rtl_small/processed_entries.jsonl")]
last=rows[-15:]
ok=sum(1 for r in last if r.get("tb_result"))
print("Последние 15 с тестбенчами:", ok, "/", len(last))
PY
```

### Интерпретация результатов
- **Поле `tb_result`:** содержит текст, записанный `$fdisplay` в тестбенче. Обычно включает ≥100 тест-векторов.
- **Поле `tb`:** полный код тестбенча. Если `null`, все попытки сгенерировать TB провалились — перезапустите скрипт.
- **`run.log`:** сводка содержит строки вроде:
  ```
  Processed: 15/15
  Successful simulations: 10
  Deterministic implementations: 0
  ```
  Проверка «deterministic» означает второй прогон тестбенча; если результаты различаются, система отмечает это, но запись сохраняется.

## 7. Мини-демо через `demo_lmstudio_verilog.py`
Чтобы быстро проверить связку, воспользуйтесь скриптом:
```bash
python demo_lmstudio_verilog.py
cat demo_module.v
```
Он отправляет запрос в LM Studio, получает простой модуль `simple_counter` и сохраняет Verilog в `demo_module.v`. Можно проверить синтаксис:
```bash
iverilog -t null demo_module.v
```

## 8. Расширенные сценарии
### Обучение (SFT/GRPO)
- **SFT:** Используйте конфиги в `SFT/LLamaFactory` или перенесите данные в `SFT/open-r1` и запустите `run_rtl_training.sh`.
- **GRPO:** Скопируйте `GRPO/verilog_rewards_tb.py` и `verilog_train_tb.py` в `open_r1`, создайте `verilog_recipe/`, затем запустите через Accelerate (см. пример в README).

### Кастомные конвейеры
- Укажите свой файл в `VERIREASON_INPUT_FILE` (JSON или JSONL).
- Измените подсказки или поведение генерации, поправив `gen_reasoning_tb.py` (например, температуру или fallback-логику).
- Настройте `VERIREASON_API_BASE` на удаленный inference-сервер или OpenAI.

## 9. Устранение неполадок
| Проблема | Причина | Решение |
|----------|---------|---------|
| `Input dataset not found` | Переменная `VERIREASON_INPUT_FILE` не задана или путь неверен | Укажите абсолютный путь (export) |
| `Simulation failed` | Неверный тестбенч | Скрипт сам переходит к следующей попытке; при необходимости увеличьте `MAX_RETRIES` или изучите поле `tb` |
| `Module name not detected` | Выходной код не содержит `module ...` | Убедитесь, что `output` содержит корректное определение модуля |
| Длительное выполнение | Медленные ответы LLM или iverilog | Уменьшите датасет (например, `sample_15`) или настройте генерацию в LM Studio |

## 10. Шаги для начала
1. Установить зависимости (Python + Icarus Verilog).
2. Настроить LM Studio (или другой endpoint), экспортировать `VERIREASON_*`.
3. Разместить датасеты в `datasets/`.
4. Запустить `python gen_reasoning_tb.py`.
5. Проверить `runs/<имя>/run.log` и `processed_entries.jsonl`.
6. При необходимости продолжить с обучением из каталогов `SFT/` и `GRPO/`.

Следуя этим шагам, вы сможете экспериментировать с генерацией, проверкой и обучением Verilog RTL при помощи VeriReason.
