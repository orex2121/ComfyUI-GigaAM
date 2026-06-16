import os
import sys
import folder_paths
import tempfile
import numpy as np
import math
import inspect
import threading
import types
import copy
import re

import torch
import torchaudio
import soundfile as sf

# --- 1. ОПРЕДЕЛЯЕМ ПАПКУ КЭША ИЗ ПЕРЕМЕННЫХ СРЕДЫ (КАК НА ВАШЕМ СКРИНШОТЕ) ---
hf_cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
if not hf_cache_dir:
    hf_cache_dir = os.path.join(folder_paths.models_dir, "audio_encoders")

os.environ["HF_HOME"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
os.environ["TORCH_HOME"] = hf_cache_dir

# Папка специально для обхода хардкода GigaAM внутри вашего кэша
TARGET_GIGAAM_DIR = os.path.join(hf_cache_dir, "gigaam_models")
os.makedirs(TARGET_GIGAAM_DIR, exist_ok=True)

print(f"\n[GigaAM Node] Базовый кэш HuggingFace: {hf_cache_dir}")
print(f"[GigaAM Node] Модели GigaAM будут скачаны в: {TARGET_GIGAAM_DIR}\n")

# --- 2. ГЛУБОКИЙ ПАТЧ ДЛЯ GIGAAM ---
# Библиотека gigaam жестко прописывает путь ~/.cache/gigaam.
# Этот патч перехватывает этот путь и подменяет на вашу папку
orig_expanduser = os.path.expanduser
def patched_expanduser(path):
    res = orig_expanduser(path)
    if '.cache' in res.replace('\\', '/') and 'gigaam' in res.lower():
        return TARGET_GIGAAM_DIR
    return res
os.path.expanduser = patched_expanduser

# --- 3. ПАТЧ HUGGINGFACE HUB ---
try:
    import huggingface_hub
    
    orig_dl = huggingface_hub.hf_hub_download
    orig_snapshot = huggingface_hub.snapshot_download
    
    if getattr(orig_dl, "_is_patched", False) is False:
        def patched_dl(*args, **kwargs):
            if 'use_auth_token' in kwargs:
                kwargs['token'] = kwargs.pop('use_auth_token')
            
            # Перехват локальной директории GigaAM
            if kwargs.get('local_dir') and '.cache' in kwargs['local_dir'].replace('\\', '/') and 'gigaam' in kwargs['local_dir']:
                kwargs['local_dir'] = TARGET_GIGAAM_DIR
            if kwargs.get('cache_dir') and '.cache' in kwargs['cache_dir'].replace('\\', '/') and 'gigaam' in kwargs['cache_dir']:
                kwargs['cache_dir'] = TARGET_GIGAAM_DIR
            
            args_list = list(args)
            repo_id_val = args_list[0] if len(args_list) > 0 else kwargs.get('repo_id')
            filename_val = args_list[1] if len(args_list) > 1 else kwargs.get('filename')
            
            if isinstance(repo_id_val, str) and filename_val is not None:
                changed = False
                if repo_id_val.startswith('$model/'):
                    sub = repo_id_val.split('/', 1)[1]
                    repo_id_val = 'pyannote/speaker-diarization-community-1'
                    filename_val = f"{sub}/{filename_val}"
                    changed = True
                elif repo_id_val.count('/') > 1:
                    parts = repo_id_val.split('/')
                    repo_id_val = f"{parts[0]}/{parts[1]}"
                    sub = '/'.join(parts[2:])
                    filename_val = f"{sub}/{filename_val}"
                    changed = True
                    
                if changed:
                    if len(args_list) > 0: args_list[0] = repo_id_val
                    else: kwargs['repo_id'] = repo_id_val
                    if len(args_list) > 1: args_list[1] = filename_val
                    else: kwargs['filename'] = filename_val
                        
            return orig_dl(*args_list, **kwargs)

        patched_dl._is_patched = True
        
        def patched_snapshot(*args, **kwargs):
            if 'use_auth_token' in kwargs:
                kwargs['token'] = kwargs.pop('use_auth_token')
            if kwargs.get('local_dir') and '.cache' in kwargs.get('local_dir', '').replace('\\', '/') and 'gigaam' in kwargs['local_dir']:
                kwargs['local_dir'] = TARGET_GIGAAM_DIR
            if kwargs.get('cache_dir') and '.cache' in kwargs.get('cache_dir', '').replace('\\', '/') and 'gigaam' in kwargs['cache_dir']:
                kwargs['cache_dir'] = TARGET_GIGAAM_DIR
            return orig_snapshot(*args, **kwargs)
            
        patched_snapshot._is_patched = True

        huggingface_hub.hf_hub_download = patched_dl
        if hasattr(huggingface_hub, 'file_download'):
            huggingface_hub.file_download.hf_hub_download = patched_dl
        huggingface_hub.snapshot_download = patched_snapshot
        
        for mod_name, mod in list(sys.modules.items()):
            if hasattr(mod, 'hf_hub_download'):
                setattr(mod, 'hf_hub_download', patched_dl)
            if hasattr(mod, 'snapshot_download'):
                setattr(mod, 'snapshot_download', patched_snapshot)
except Exception as e:
    print(f"[GigaAM Node] Ошибка перенаправления HF: {e}")
# --------------------------------------------------------------------------

torch.hub.set_dir(hf_cache_dir)

# --- ИСПРАВЛЕНИЕ БЕСКОНЕЧНОЙ РЕКУРСИИ И СБОЕВ ПРИ ИМПОРТЕ SPEECHBRAIN ---
try:
    import speechbrain.utils.importutils
    if not hasattr(speechbrain.utils.importutils.LazyModule, '_sb_patch_applied'):
        _sb_lock = threading.local()
        _orig_getattr = speechbrain.utils.importutils.LazyModule.__getattr__
        
        def _patched_getattr(self, attr):
            if getattr(_sb_lock, 'locked', False):
                raise AttributeError(attr)
            
            _sb_lock.locked = True
            try:
                return _orig_getattr(self, attr)
            except ImportError:
                raise AttributeError(attr)
            finally:
                _sb_lock.locked = False
                
        speechbrain.utils.importutils.LazyModule.__getattr__ = _patched_getattr
        speechbrain.utils.importutils.LazyModule._sb_patch_applied = True
except Exception:
    pass
# ------------------------------------------------------------------------

# --- ЗАГЛУШКИ ДЛЯ СОВМЕСТИМОСТИ PYANNOTE И НОВОГО TORCHAUDIO ---
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda backend: None

if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"

if 'torchaudio.backend' not in sys.modules:
    sys.modules['torchaudio.backend'] = types.ModuleType('torchaudio.backend')

if 'torchaudio.backend.common' not in sys.modules:
    common_mod = types.ModuleType('torchaudio.backend.common')
    common_mod.AudioMetaData = getattr(torchaudio, 'AudioMetaData', type('AudioMetaData', (object,), {}))
    sys.modules['torchaudio.backend.common'] = common_mod
# ---------------------------------------------------------------

# --- ЗАГЛУШКА ДЛЯ СОВМЕСТИМОСТИ PYANNOTE И НОВОГО NUMPY 2.0+ ---
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'NAN'):
    np.NAN = np.nan
# ---------------------------------------------------------------

# 3. ТОЛЬКО ТЕПЕРЬ (после патча) ИМПОРТИРУЕМ GIGAAM И PYANNOTE
try:
    import gigaam
except ImportError:
    print("\n[GigaAM Node] ВНИМАНИЕ: Библиотека GigaAM не установлена!")
    print("[GigaAM Node] Пожалуйста, выполните установку GigaAM\n")

try:
    from pyannote.audio import Pipeline
    
    # --- ИСПРАВЛЕНИЕ ОШИБКИ WEIGHTS_ONLY В PYTORCH 2.6+ И ВЫШЕ ---
    try:
        if hasattr(torch.serialization, 'add_safe_globals'):
            from pyannote.audio.core.task import Specifications
            torch.serialization.add_safe_globals([Specifications])
            
            try:
                from pyannote.audio.core.task import Resolution
                torch.serialization.add_safe_globals([Resolution])
            except Exception:
                pass
                
            try:
                from pyannote.audio.core.task import Problem
                torch.serialization.add_safe_globals([Problem])
            except Exception:
                pass
    except Exception as e:
        pass
    # -------------------------------------------------------------
    
    # --- ПАТЧ ДЛЯ ОШИБКИ KeyError: 'torch' В PYANNOTE 3.1.1 ---
    try:
        from pyannote.audio.core.model import Model
        if not getattr(Model.on_load_checkpoint, "_is_patched", False):
            orig_on_load_checkpoint = Model.on_load_checkpoint
            def patched_on_load_checkpoint(self, checkpoint):
                if "pyannote.audio" not in checkpoint:
                    checkpoint["pyannote.audio"] = {}
                if "versions" not in checkpoint["pyannote.audio"]:
                    checkpoint["pyannote.audio"]["versions"] = {}
                if "torch" not in checkpoint["pyannote.audio"]["versions"]:
                    checkpoint["pyannote.audio"]["versions"]["torch"] = "2.0.0"
                if "pyannote.audio" not in checkpoint["pyannote.audio"]["versions"]:
                    checkpoint["pyannote.audio"]["versions"]["pyannote.audio"] = "3.1.1"
                
                return orig_on_load_checkpoint(self, checkpoint)
            
            patched_on_load_checkpoint._is_patched = True
            Model.on_load_checkpoint = patched_on_load_checkpoint
    except Exception:
        pass
    # ----------------------------------------------------------
            
    # --- ИСПРАВЛЕНИЕ ОШИБКИ УСТАРЕВШИХ ПАРАМЕТРОВ КОНФИГА (plda) И КЛАСТЕРИЗАЦИИ ---
    try:
        from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
        if not getattr(SpeakerDiarization.__init__, "_is_patched", False):
            orig_sd_init = SpeakerDiarization.__init__
            def patched_sd_init(self, *args, **kwargs):
                kwargs.pop('plda', None)
                if kwargs.get('clustering') not in ['AgglomerativeClustering', 'OracleClustering']:
                    kwargs['clustering'] = 'AgglomerativeClustering'
                orig_sd_init(self, *args, **kwargs)
            patched_sd_init._is_patched = True
            SpeakerDiarization.__init__ = patched_sd_init
    except Exception:
        pass
    # -------------------------------------------------------------------------------

    # --- ПАТЧ ДЛЯ ОШИБКИ ИНСТАНЦИРОВАНИЯ (Fa / VBx гиперпараметры) ---
    try:
        from pyannote.pipeline import Pipeline as BasePipeline
        if not getattr(BasePipeline.instantiate, "_is_patched", False):
            orig_instantiate = BasePipeline.instantiate
            def patched_instantiate(self, params: dict):
                safe_params = copy.deepcopy(params)
                if "clustering" in safe_params and isinstance(safe_params["clustering"], dict):
                    if any(k in safe_params["clustering"] for k in ["Fa", "Fb", "loop_p", "loop_probabilities"]):
                        safe_params["clustering"] = {
                            "method": "centroid",
                            "min_cluster_size": 12,
                            "threshold": 0.7045654963945799
                        }
                return orig_instantiate(self, safe_params)
            
            patched_instantiate._is_patched = True
            BasePipeline.instantiate = patched_instantiate
    except Exception:
        pass
    # -----------------------------------------------------------------

except ImportError:
    Pipeline = None
    print("\n[GigaAM Node] ВНИМАНИЕ: Библиотека pyannote.audio не установлена!")
    print("[GigaAM Node] Диаризация спикеров работать не будет. Выполните: pip install pyannote.audio\n")


class GigaAM_Transcription:
    DESCRIPTION = """
    Узел для транскрипции аудио с использованием моделей GigaAM и диаризации Pyannote.

    Настройки:
    - для скачивания моделей зарегистрируйтесь на huggingface и создайте hf token для чтения.
    - model_name: Выбор модели GigaAM. Рекомендуется v3_e2e_rnnt.
    - longform: Включите для аудио более 20 сек или всегда держите включенным. 
    - hf_token: Токен Hugging Face (нужен только при скачивании модели).
    - word_timestamps: Вывод временных меток.
    - sentences_per_interval: Вывод по N предложений в строку. Приоритетнее чем words_per_interval.
    - words_per_interval: Вывод по N слов в строку. Если sentences_per_interval=0 и words_per_interval=0 то разбивается по предложениям (по умолчанию).
    - chunk_tokens: Разбить весь итоговый текст на блоки по количеству токенов (полезно для LLM). Разбиение происходит только в конце предложений. Результат выводится в text_batch.
    - highlight_words: Вывод субтитров в формате SRT с подсветкой текущего слова <u> (караоке).
    - speaker_diarization: Включает определение спикеров.
    - min_speakers / max_speakers: лимиты спикеров.
    - speaker_format: формат обозначения спикеров.
    """

    def __init__(self):
        self.audio_models_dir = hf_cache_dir

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (["v3_e2e_rnnt", "v3_e2e_ctc", "v3_rnnt", "v3_ctc", "v2_rnnt", "v2_ctc", "v1_rnnt", "v1_ctc"], {"default": "v3_e2e_rnnt"}),
                "longform": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No"}),
                "hf_token": ("STRING", {
                    "default": "", 
                    "placeholder": "Оставьте пустым, если модель скачана или есть ENV токен", 
                    "multiline": False
                }),
                "word_timestamps": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "sentences_per_interval": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "words_per_interval": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "chunk_tokens": ("INT", {"default": 3072, "min": 0, "max": 100000, "step": 256}),
                "highlight_words": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "speaker_diarization": ("BOOLEAN", {"default": False, "label_on": "On", "label_off": "Off"}),
                "min_speakers": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "max_speakers": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "speaker_format": (["SPEAKER A: SPEAKER B: SPEAKER C:", "[Speaker_1]: [Speaker_2]: [Speaker_3]:"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "text_batch")
    OUTPUT_IS_LIST = (False, True) # text_batch выводится как список строк (batch)
    FUNCTION = "transcribe_audio"
    CATEGORY = "Audio/GigaAM"

    def split_audio_by_silence(self, waveform, sample_rate, frame_duration_ms=30, min_silence_duration_ms=500, silence_threshold=0.01):
        if len(waveform.shape) > 1:
            mono_waveform = waveform.mean(dim=0)
        else:
            mono_waveform = waveform

        frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
        
        # Получаем энергию фреймов
        energy = torch.nn.functional.avg_pool1d(
            mono_waveform.abs().unsqueeze(0).unsqueeze(0), 
            kernel_size=frame_length, 
            stride=frame_length
        ).squeeze()
        
        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
            
        # Улучшенный динамический порог тишины (адаптируется к громкости аудио)
        max_energy = torch.max(energy).item()
        dynamic_silence_threshold = max(max_energy * 0.02, 0.001) # 2% от максимальной громкости
        
        is_speech = (energy > dynamic_silence_threshold).tolist()
        
        segments = []
        current_start = None
        silence_counter = 0
        
        # Защита: жесткий лимит длины куска ~20 секунд
        max_chunk_frames = int(20.0 * sample_rate)
        
        for i, speech_flag in enumerate(is_speech):
            current_time_frames = i * frame_length
            
            if speech_flag:
                if current_start is None:
                    current_start = current_time_frames
                silence_counter = 0
                
                # Если текущий сегмент речи превысил лимит в 20 секунд, безопасно отрезаем его и продолжаем
                if (current_time_frames - current_start) >= max_chunk_frames:
                    segments.append((current_start, current_time_frames))
                    current_start = current_time_frames
            else:
                if current_start is not None:
                    silence_counter += 1
                    if silence_counter >= min_silence_frames:
                        segments.append((current_start, current_time_frames))
                        current_start = None
                        silence_counter = 0
        
        # Если аудио закончилось, а кусок речи еще не закрыт - добавляем его
        if current_start is not None:
            if mono_waveform.shape[0] - current_start > int(0.1 * sample_rate):
                segments.append((current_start, mono_waveform.shape[0]))
            
        return segments

    def generate_karaoke_srt(self, words_list, start_index=1, speaker_prefix=""):
        srt_lines = []
        srt_index = start_index

        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        sentence_words = [w['text'] for w in words_list]

        for i, w in enumerate(words_list):
            start_time = format_srt_time(w['start'])
            
            if i < len(words_list) - 1:
                end_time = format_srt_time(words_list[i+1]['start'])
            else:
                end_time = format_srt_time(w['end'])

            highlighted_sentence = []
            for j, text in enumerate(sentence_words):
                if j == i:
                    highlighted_sentence.append(f"<u>{text}</u>")
                else:
                    highlighted_sentence.append(text)

            srt_lines.append(str(srt_index))
            srt_lines.append(f"{start_time} --> {end_time}")
            
            line_text = speaker_prefix + " ".join(highlighted_sentence)
            srt_lines.append(line_text)
            srt_lines.append("")
            srt_index += 1

        return "\n".join(srt_lines), srt_index

    def get_speaker_for_time(self, diarization, time_sec):
        if diarization is None:
            return ""
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= time_sec <= turn.end:
                return speaker
        return ""

    def format_speaker(self, speaker_label, format_type):
        if not speaker_label:
            return ""
        
        try:
            num = int(speaker_label.split("_")[-1])
        except ValueError:
            num = 0
            
        if format_type == "SPEAKER A: SPEAKER B: SPEAKER C:":
            return f"SPEAKER {chr(65 + num)}: "
        elif format_type == "[Speaker_1]: [Speaker_2]: [Speaker_3]:":
            return f"[Speaker_{num + 1}]: "
            
        return f"{speaker_label}: "

    def transcribe_audio(self, audio, model_name, longform, hf_token, word_timestamps, sentences_per_interval, words_per_interval, chunk_tokens, highlight_words, speaker_diarization, min_speakers, max_speakers, speaker_format):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(0)

        # --- БЕЗОПАСНАЯ ИНИЦИАЛИЗАЦИЯ ТОКЕНА HF ---
        actual_hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
        if not actual_hf_token:
            actual_hf_token = hf_token.strip()

        # --- БЛОК ДИАРИЗАЦИИ PYANNOTE ---
        diarization = None
        if speaker_diarization:
            if Pipeline is None:
                raise Exception("pyannote.audio не установлена! Выполните `pip install pyannote.audio` или отключите Speaker Diarization.")
            
            diarization_model_id = "pyannote/speaker-diarization-community-1"
            pipeline = None
            
            print("[GigaAM Node] Проверка локальной модели диаризации...")
            orig_offline = os.environ.get("HF_HUB_OFFLINE", None)
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            try:
                pipeline = Pipeline.from_pretrained(diarization_model_id, cache_dir=self.audio_models_dir)
                print("[GigaAM Node] Модель диаризации найдена локально, загружена без обращения к сети.")
            except Exception as offline_e:
                if orig_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = orig_offline
                else:
                    del os.environ["HF_HUB_OFFLINE"]
                    
                print("[GigaAM Node] Локальная модель не найдена в папке. Начинаем скачивание...")
                if not actual_hf_token:
                    raise Exception(f"Для скачивания модели {diarization_model_id} необходим HF Token!")
                
                try:
                    pipeline = Pipeline.from_pretrained(diarization_model_id, use_auth_token=actual_hf_token, cache_dir=self.audio_models_dir)
                except Exception as online_e:
                    raise Exception(f"Ошибка загрузки Pyannote: {online_e}")
            finally:
                if "HF_HUB_OFFLINE" in os.environ:
                    if orig_offline is not None:
                        os.environ["HF_HUB_OFFLINE"] = orig_offline
                    else:
                        del os.environ["HF_HUB_OFFLINE"]
            
            if pipeline is not None:
                if torch.cuda.is_available():
                    pipeline.to(torch.device("cuda"))
                
                print("[GigaAM Node] Запуск анализа спикеров...")
                d_wave = waveform.clone()
                if len(d_wave.shape) == 1:
                    d_wave = d_wave.unsqueeze(0)
                if d_wave.shape[0] > 1:
                    d_wave = d_wave.mean(dim=0, keepdim=True)
                
                audio_in_memory = {"waveform": d_wave, "sample_rate": sample_rate}
                diarization = pipeline(audio_in_memory, min_speakers=min_speakers, max_speakers=max_speakers)
                print("[GigaAM Node] Анализ спикеров завершен.")

        need_word_level = word_timestamps or sentences_per_interval > 0 or words_per_interval > 0 or highlight_words or speaker_diarization

        print(f"[GigaAM Node] Загрузка модели транскрипции {model_name}...")
        model = gigaam.load_model(model_name)
        print("[GigaAM Node] Начинаю транскрипцию текста...")
        
        final_text = []
        all_words_flat = [] 
        sentences_longform = []
        
        if longform:
            segments_indices = self.split_audio_by_silence(waveform, sample_rate)
            
            for start_idx, end_idx in segments_indices:
                chunk_waveform = waveform[:, start_idx:end_idx]
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    audio_data = chunk_waveform.cpu().numpy().T
                    sf.write(tmp_path, audio_data, sample_rate)
                    
                    if need_word_level:
                        result = model.transcribe(tmp_path, word_timestamps=True)
                        chunk_start_time = start_idx / sample_rate
                        chunk_words = []
                        for w in result.words:
                            chunk_words.append({
                                'start': w.start + chunk_start_time,
                                'end': w.end + chunk_start_time,
                                'text': w.text
                            })
                        
                        all_words_flat.extend(chunk_words)
                        if highlight_words and chunk_words:
                            sentences_longform.append(chunk_words)
                    else:
                        result = model.transcribe(tmp_path)
                        text = result if isinstance(result, str) else result.text
                        if text and text.strip():
                            final_text.append(text)
                except Exception as e:
                     print(f"[GigaAM Node] Ошибка транскрипции чанка: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                audio_data = waveform.cpu().numpy().T
                sf.write(tmp_path, audio_data, sample_rate)
                
                if need_word_level:
                    result = model.transcribe(tmp_path, word_timestamps=True)
                    chunk_words = [{'start': w.start, 'end': w.end, 'text': w.text} for w in result.words]
                    all_words_flat.extend(chunk_words)
                    if highlight_words and chunk_words:
                        sentences_longform.append(chunk_words)
                else:
                    result = model.transcribe(tmp_path)
                    text = result if isinstance(result, str) else result.text
                    if text and text.strip():
                        final_text.append(text)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # --- ПОСТОБРАБОТКА И ФОРМАТИРОВАНИЕ ---
        output_text = ""
        text_batch = []
        
        if highlight_words:
            final_srt = []
            srt_index = 1
            for sent in sentences_longform:
                spk_prefix = ""
                if speaker_diarization and sent:
                    mid_time = (sent[0]['start'] + sent[-1]['end']) / 2.0
                    spk = self.get_speaker_for_time(diarization, mid_time)
                    spk_prefix = self.format_speaker(spk, speaker_format)

                srt_text, srt_index = self.generate_karaoke_srt(sent, srt_index, spk_prefix)
                final_srt.append(srt_text)
                
            output_text = "\n".join(final_srt).strip()
            if output_text:
                text_batch = [output_text]

        elif need_word_level and all_words_flat:
            # Условие 1: Разбивка по количеству ПРЕДЛОЖЕНИЙ (если включены таймкоды и параметр > 0)
            if word_timestamps and sentences_per_interval > 0:
                current_speaker = None
                current_text = []
                current_start = None
                current_end = None
                sentence_count = 0

                for w in all_words_flat:
                    mid_time = (w['start'] + w['end']) / 2.0
                    spk_prefix = ""
                    if speaker_diarization:
                        spk = self.get_speaker_for_time(diarization, mid_time)
                        spk_prefix = self.format_speaker(spk, speaker_format)
                    
                    speaker_changed = (current_speaker is not None) and (spk_prefix != current_speaker)
                    
                    if speaker_changed:
                        if current_text:
                            text_str = " ".join(current_text)
                            start_str = gigaam.format_time(current_start)
                            end_str = gigaam.format_time(current_end)
                            final_text.append(f"[{start_str} - {end_str}]: {current_speaker}{text_str}")
                        
                        current_speaker = spk_prefix
                        current_text = [w['text']]
                        current_start = w['start']
                        current_end = w['end']
                        sentence_count = 1 if w['text'].strip().endswith(('.', '!', '?')) else 0
                    else:
                        if current_speaker is None:
                            current_speaker = spk_prefix
                            current_start = w['start']
                        
                        current_text.append(w['text'])
                        current_end = w['end']
                        
                        if w['text'].strip().endswith(('.', '!', '?')):
                            sentence_count += 1
                            
                        if sentence_count >= sentences_per_interval:
                            text_str = " ".join(current_text)
                            start_str = gigaam.format_time(current_start)
                            end_str = gigaam.format_time(current_end)
                            final_text.append(f"[{start_str} - {end_str}]: {current_speaker}{text_str}")
                            
                            current_text = []
                            current_speaker = None
                            current_start = None
                            current_end = None
                            sentence_count = 0

                if current_text:
                    text_str = " ".join(current_text)
                    start_str = gigaam.format_time(current_start)
                    end_str = gigaam.format_time(current_end)
                    final_text.append(f"[{start_str} - {end_str}]: {current_speaker}{text_str}")

            # Условие 2: Разбивка по количеству СЛОВ (если включены таймкоды и слова > 0)
            elif word_timestamps and words_per_interval > 0:
                for i in range(0, len(all_words_flat), words_per_interval):
                    group = all_words_flat[i:i + words_per_interval]
                    text_str = " ".join([w['text'] for w in group])
                    
                    spk_prefix = ""
                    if speaker_diarization:
                        mid_time = (group[0]['start'] + group[-1]['end']) / 2.0
                        spk = self.get_speaker_for_time(diarization, mid_time)
                        spk_prefix = self.format_speaker(spk, speaker_format)

                    start_str = gigaam.format_time(group[0]['start'])
                    end_str = gigaam.format_time(group[-1]['end'])
                    final_text.append(f"[{start_str} - {end_str}]: {spk_prefix}{text_str}")
            
            # Условие 3: Резервный вариант (Оба = 0, или выключены таймкоды, но включена диаризация)
            else:
                current_speaker = None
                current_text = []
                current_start = None
                current_end = None

                for w in all_words_flat:
                    mid_time = (w['start'] + w['end']) / 2.0
                    spk_prefix = ""
                    if speaker_diarization:
                        spk = self.get_speaker_for_time(diarization, mid_time)
                        spk_prefix = self.format_speaker(spk, speaker_format)
                    
                    is_sentence_end = False
                    if current_text:
                        if current_text[-1].strip().endswith(('.', '!', '?')):
                            is_sentence_end = True
                    
                    should_break = (spk_prefix != current_speaker) or is_sentence_end
                    
                    if current_speaker is None:
                        current_speaker = spk_prefix
                        current_text.append(w['text'])
                        current_start = w['start']
                        current_end = w['end']
                    elif not should_break:
                        current_text.append(w['text'])
                        current_end = w['end']
                    else:
                        text_str = " ".join(current_text)
                        if word_timestamps:
                            start_str = gigaam.format_time(current_start)
                            end_str = gigaam.format_time(current_end)
                            final_text.append(f"[{start_str} - {end_str}]: {current_speaker}{text_str}")
                        else:
                            final_text.append(f"{current_speaker}{text_str}")
                        
                        current_speaker = spk_prefix
                        current_text = [w['text']]
                        current_start = w['start']
                        current_end = w['end']

                if current_text:
                    text_str = " ".join(current_text)
                    if word_timestamps:
                        start_str = gigaam.format_time(current_start)
                        end_str = gigaam.format_time(current_end)
                        final_text.append(f"[{start_str} - {end_str}]: {current_speaker}{text_str}")
                    else:
                        final_text.append(f"{current_speaker}{text_str}")

        # === ПРИМЕНЕНИЕ CHUNK_TOKENS И СБОРКА ИТОГОВОГО ТЕКСТА ===
        output_text = "\n".join(final_text).strip()
        
        if not highlight_words:
            if chunk_tokens > 0 and output_text:
                # Разбиваем текст на предложения (разделители: точки, вопросительные и восклицательные знаки)
                # re.split оставит разделители в массиве, чтобы мы могли собрать предложения обратно
                parts = re.split(r'(?<=[.!?])(\s+)', output_text)
                sentences = []
                temp_str = ""
                
                for i, part in enumerate(parts):
                    temp_str += part
                    # Каждый второй элемент — это пробел или перевод строки (разделитель)
                    if i % 2 == 1 or i == len(parts) - 1:
                        if temp_str:
                            sentences.append(temp_str)
                            temp_str = ""
                
                current_chunk = ""
                current_token_count = 0
                
                for sentence in sentences:
                    # Эвристика: 1 токен LLM ≈ 3 символа
                    est_tokens = max(1, len(sentence) // 3)
                    
                    if current_token_count + est_tokens > chunk_tokens:
                        if current_chunk.strip():
                            # Сохраняем собранный чанк, затем начинаем новый
                            text_batch.append(current_chunk.strip())
                            current_chunk = sentence
                            current_token_count = est_tokens
                        else:
                            # Случай, когда одно длинное предложение превышает лимит.
                            # Не разрываем его, а смещаем разбиение на конец предложения
                            text_batch.append(sentence.strip())
                            current_chunk = ""
                            current_token_count = 0
                    else:
                        # Добавляем предложение в текущий чанк
                        current_chunk += sentence
                        current_token_count += est_tokens
                        
                # Добавляем остаток текста в последний чанк
                if current_chunk.strip():
                    text_batch.append(current_chunk.strip())
            else:
                # Если разбиение отключено (chunk_tokens = 0)
                if output_text:
                    text_batch = [output_text]

        print("[GigaAM Node] Транскрипция завершена.")
        return (output_text, text_batch)

NODE_CLASS_MAPPINGS = {"GigaAM_Transcription": GigaAM_Transcription}
NODE_DISPLAY_NAME_MAPPINGS = {"GigaAM_Transcription": "GigaAM Speech to Text"}