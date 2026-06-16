import re
import ast

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

        for line in lines:
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