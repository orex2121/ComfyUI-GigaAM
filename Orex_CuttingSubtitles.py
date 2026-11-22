import re
import ast


SRT_TIME_PATTERN = re.compile(
    r"^(\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def _srt_time_to_orex(value):
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds, milliseconds = rest.split(".")
    total_minutes = int(hours) * 60 + int(minutes)
    return f"{total_minutes:02d}:{int(seconds):02d}:{int(milliseconds) // 10:02d}"

class Orex_CuttingSubtitles:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitles_text": ("STRING", {"multiline": True, "default": "[00:00:28 - 00:02:12]: Hello everyone"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("subs_with_timestamps (batch)", "clean_text (batch)")
    OUTPUT_IS_LIST = (True, True) 
    FUNCTION = "process_subtitles"
    CATEGORY = "Orex Nodes 🛠️"

    def process_subtitles(self, subtitles_text):
        subs_with_timestamps = []
        clean_text = []

        # 1. Если пришел настоящий список, клеим его
        if isinstance(subtitles_text, list):
            subtitles_text = "\n".join(str(item) for item in subtitles_text)
        else:
            subtitles_text = str(subtitles_text)

        # 2. Очистка от строковых массивов: если LLM выдала текст в виде "['текст\nтекст']"
        subtitles_text = subtitles_text.strip()
        if subtitles_text.startswith('[') and subtitles_text.endswith(']'):
            try:
                # Пытаемся безопасно распарсить строку как Python-список
                parsed = ast.literal_eval(subtitles_text)
                if isinstance(parsed, list):
                    subtitles_text = "\n".join(str(item) for item in parsed)
            except Exception:
                pass # Если это просто текст в квадратных скобках, идем дальше
        
        # 3. Чиним фейковые переносы строк (если LLM отдала \n как текст)
        subtitles_text = subtitles_text.replace('\\n', '\n')

        # 4. Теперь разбиваем текст на реальные строки
        lines = [line.strip() for line in subtitles_text.split('\n') if line.strip()]

        # 5. Пуленепробиваемая регулярка. 
        # .*? в самом начале игнорирует любой мусор (например, оставшиеся кавычки) до таймкода
        pattern = re.compile(r"^.*?(\[[0-9:\s-]+\])[\s:]*(.*)$")

        line_index = 0
        while line_index < len(lines):
            line = lines[line_index]

            # SRT cue numbers are service lines and are not part of the subtitle text.
            if (
                line.isdigit()
                and line_index + 1 < len(lines)
                and SRT_TIME_PATTERN.match(lines[line_index + 1])
            ):
                line_index += 1
                line = lines[line_index]

            srt_match = SRT_TIME_PATTERN.match(line)
            if srt_match:
                line_index += 1
                cue_text = []
                while line_index < len(lines):
                    next_line = lines[line_index]
                    if SRT_TIME_PATTERN.match(next_line) or (
                        next_line.isdigit()
                        and line_index + 1 < len(lines)
                        and SRT_TIME_PATTERN.match(lines[line_index + 1])
                    ):
                        break
                    cue_text.append(next_line)
                    line_index += 1

                text = " ".join(cue_text).rstrip("'\"]")
                start = _srt_time_to_orex(srt_match.group(1))
                end = _srt_time_to_orex(srt_match.group(2))
                subs_with_timestamps.append(f"[{start} - {end}]: {text}")
                clean_text.append(text)
                continue

            match = pattern.match(line)
            if match:
                timestamp = match.group(1)
                text = match.group(2)
                
                # Отсекаем мусор от LLM в конце самого текста (лишние кавычки или скобки)
                text = text.rstrip("'\"]")
                
                # Собираем обратно таймкод и текст для первого выхода
                subs_with_timestamps.append(f"{timestamp}: {text}")
                # Только чистый текст для второго выхода (генератора)
                clean_text.append(text)
            else:
                # Если строка совсем не содержит таймкод, пропускаем её без изменений
                subs_with_timestamps.append(line)
                clean_text.append(line)

            line_index += 1

        # Подстраховка от пустых списков
        if not subs_with_timestamps:
            subs_with_timestamps = [""]
            clean_text = [""]

        return (subs_with_timestamps, clean_text)


NODE_CLASS_MAPPINGS = {
    "Orex_CuttingSubtitles": Orex_CuttingSubtitles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orex_CuttingSubtitles": "Orex Cutting Subtitles ✂️"
}
