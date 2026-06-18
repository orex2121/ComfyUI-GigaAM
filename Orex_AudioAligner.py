import torch
import re
import numpy as np

try:
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter
    HAS_AUDIOTSM = True
except ImportError:
    HAS_AUDIOTSM = False
    print("\n[Orex] ОШИБКА: Не установлена библиотека audiotsm. Выполните: pip install audiotsm\n")

class Orex_AudioAligner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_batch": ("AUDIO",),
                "timestamps_batch": ("STRING",),
            }
        }

    INPUT_IS_LIST = True 
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("aligned_audio",)
    FUNCTION = "align_audio"
    CATEGORY = "Orex Nodes 🛠️"

    def parse_time(self, time_str):
        parts = time_str.split(':')
        if len(parts) == 3:
            m = int(parts[0])
            s = int(parts[1])
            cs = int(parts[2]) 
            return m * 60 + s + cs / 100.0
        elif len(parts) == 4:
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2])
            cs = int(parts[3])
            return h * 3600 + m * 60 + s + cs / 100.0
        return 0.0

    def align_audio(self, audio_batch, timestamps_batch):
        if not HAS_AUDIOTSM:
            raise RuntimeError("Библиотека audiotsm не установлена. Выполните pip install audiotsm")

        print(f"\n[Orex Aligner] Получено аудиодорожек: {len(audio_batch)}")
        print(f"[Orex Aligner] Получено таймкодов: {len(timestamps_batch)}")

        time_pattern = re.compile(r"^\[([0-9:]+)\s*-\s*([0-9:]+)\]")
        parsed_data = []
        max_end_time = 0.0

        for i, text_line in enumerate(timestamps_batch):
            match = time_pattern.match(text_line)
            if match and i < len(audio_batch):
                start_time = self.parse_time(match.group(1))
                end_time = self.parse_time(match.group(2))
                target_duration = end_time - start_time
                
                if end_time > max_end_time:
                    max_end_time = end_time
                    
                parsed_data.append({
                    "start": start_time,
                    "target_duration": target_duration,
                    "audio": audio_batch[i]
                })

        if not parsed_data:
            return (audio_batch[0],)

        sample_rate = parsed_data[0]["audio"]["sample_rate"]
        total_samples = int((max_end_time + 1.0) * sample_rate)
        final_waveform = np.zeros(total_samples, dtype=np.float32)

        for data in parsed_data:
            waveform_tensor = data["audio"]["waveform"]
            if waveform_tensor.shape[1] > 1:
                waveform_tensor = waveform_tensor.mean(dim=1, keepdim=True)
                
            audio_np = waveform_tensor.squeeze().numpy()
            actual_duration = len(audio_np) / sample_rate
            
            # Защита от нулевой или отрицательной длительности
            if data["target_duration"] <= 0.001:
                print(f"[Orex Aligner] Внимание: некорректная длительность целевого фрагмента ({data['target_duration']}s). Пропуск подгонки.")
                rate = 1.0
            else:
                rate = actual_duration / data["target_duration"]
                # Ограничиваем скорость, чтобы алгоритм не сошел с ума на экстремальных значениях
                rate = np.clip(rate, 0.5, 2.0) 

            print(f"[Orex Aligner] Подгонка: {actual_duration:.2f}s -> {data['target_duration']:.2f}s (Скорость: {rate:.2f}x)")

            if abs(rate - 1.0) > 0.05:
                # --- НОВЫЙ АЛГОРИТМ WSOLA ---
                # Переводим numpy массив в формат, понятный audiotsm (channels, samples)
                reader = ArrayReader(audio_np.reshape(1, -1))
                writer = ArrayWriter(channels=1)
                
                # Инициализируем WSOLA с нужной скоростью
                tsm = wsola(channels=1, speed=rate)
                tsm.run(reader, writer)
                
                # Достаем обработанный звук и сплющиваем обратно в 1D массив
                stretched_audio = writer.data.reshape(-1)
            else:
                stretched_audio = audio_np

            start_sample = int(data["start"] * sample_rate)
            end_sample = start_sample + len(stretched_audio)

            if end_sample > len(final_waveform):
                pad_length = end_sample - len(final_waveform)
                final_waveform = np.pad(final_waveform, (0, pad_length), 'constant')

            final_waveform[start_sample:end_sample] += stretched_audio

        max_val = np.max(np.abs(final_waveform))
        if max_val > 1.0:
            final_waveform = final_waveform / max_val

        final_tensor = torch.from_numpy(final_waveform).unsqueeze(0).unsqueeze(0)
        
        final_audio_dict = {
            "waveform": final_tensor,
            "sample_rate": sample_rate
        }

        print("[Orex Aligner] Сборка (WSOLA) успешно завершена!")
        # Возвращаем кортеж со словарем (исправлено)
        return (final_audio_dict,)

NODE_CLASS_MAPPINGS = {
    "Orex_AudioAligner": Orex_AudioAligner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Orex_AudioAligner": "Audio Subtitle Aligner (OreX) 🎛️"
}