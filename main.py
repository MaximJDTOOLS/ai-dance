#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DanceMaster — dance game with editor and MediaPipe Tasks API.
Single file. PyQt6 + OpenCV + MediaPipe.
"""

import sys, json, time, uuid, shutil, urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QUrl, QElapsedTimer, QSettings,
)
from PyQt6.QtGui import (
    QImage, QPixmap, QKeySequence, QShortcut, QPainter, QColor, QFont,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QLineEdit,
    QFileDialog, QMessageBox, QCheckBox, QGroupBox,
    QProgressBar, QStatusBar, QFrame, QSizePolicy, QComboBox,
    QTabWidget, QInputDialog, QDialog, QGridLayout, QDialogButtonBox,
    QStackedWidget,
)
from PyQt6.QtMultimedia import (
    QMediaPlayer, QAudioOutput, QMediaDevices, QVideoSink, QVideoFrame,
)


# ============================================================
# Safe int helper for QProgressBar (int32)
# ============================================================
def safe_pct_0_100(value) -> int:
    try:
        v = float(value)
    except Exception:
        return 0
    if v != v:
        return 0
    if v in (float('inf'), float('-inf')):
        return 0
    if v < 0:
        return 0
    if v > 100:
        return 100
    try:
        return int(v)
    except Exception:
        return 0


# ============================================================
# i18n
# ============================================================
STRINGS = {
    'en': {
        'nav_menu': 'Menu', 'nav_editor': '🎬 Editor', 'nav_play': '🎮 Play',
        'cam_group': 'Camera', 'cam_rescan': 'Rescan',
        'cam_rescan_tip': 'Rescan cameras',
        'cam_hint': 'If the camera did not open, click Rescan and pick another.',
        'cam_opening': 'Opening camera #{n}…',
        'cam_status': 'Camera #{n}: {fps:.0f} FPS',
        'cam_status_people': 'Camera #{n}: {fps:.0f} FPS  ·  people: {k}',
        'cam_error_not_found': 'Camera #{n} could not be opened.',
        'cam_error_title': 'Camera', 'cam_not_found': 'No cameras found',
        'cam_initializing': 'Initializing camera…',
        'welcome': 'Welcome!',
        'menu_info': (
            '🎬 <b>Editor</b> — two ways to create a dance:<br>'
            '&nbsp;&nbsp;• 🎥 <b>Camera recording</b> — dance to music.<br>'
            '&nbsp;&nbsp;• 📹 <b>Video reference</b> — load mp4/webm. '
            'Choose how many people dance (up to {max}).<br><br>'
            '🏷 Character names can be set when saving.<br>'
            '👑 In Play there is <b>"Auto (lead)"</b>.<br><br>'
            '🖥 <b>F11</b> — fullscreen, <b>Esc</b> — exit.<br><br>'
            '🏆 Score: <b>0 – {score_max}</b>.'
        ),
        'lang_label': 'Language:',
        'tab_cam': '🎥 Camera recording', 'tab_video': '📹 Video reference',
        'music_group': '1. Music', 'music_none': 'Not selected',
        'music_pick': '📁 Choose audio', 'music_pick_title': 'Choose audio',
        'music_filter': 'Audio (*.mp3 *.wav *.ogg *.m4a *.flac)',
        'name_group': '2. Name and video', 'name_placeholder': 'My awesome dance',
        'record_video': '🎥 Record video of my dance',
        'rec_group': '3. Dance recording',
        'rec_start': '● Start', 'rec_stop': '■ Stop',
        'rec_hint_cam': 'ℹ️ If several people are in frame, all are saved (up to {max}).',
        'rec_frames_zero': 'Frames: 0', 'rec_countdown': 'Countdown 3…',
        'rec_done': 'Done. Frames: {n}',
        'rec_done_people': 'Done. Frames: {n}  ·  max people: {k}',
        'rec_live': 'Frames: {n}  |  {t:.1f}s',
        'rec_live_people': 'Frames: {n}  |  {t:.1f}s  ·  people: {k}',
        'save_dance': '💾 Save dance', 'saved_dances_group': 'Saved dances',
        'del_dance': '🗑 Delete', 'edit_names': '🏷 Names',
        'video_info': (
            'Load a dance video (mp4, webm, mov, avi, mkv). '
            'MediaPipe tracks up to {max} people. You can specify how many '
            'people dance, or use Auto.<br><br>'
            'During Play the video plays with score overlay, no camera shown.'
        ),
        'video_pick': '📹 Choose video reference',
        'video_processing_btn': '⏳ Processing video…',
        'video_processing_status': 'Detecting poses…',
        'video_progress': 'Processing: {cur} / {total} frames ({pct}%)',
        'video_done': '✅ Done! Poses: {n}  ·  characters: {k}',
        'video_open_error': 'Could not open video file',
        'video_no_poses': 'No poses found in the video.',
        'video_processing_error': 'Video processing error',
        'video_choice_title': 'Characters in the video',
        'video_choice_prompt': 'How many people dance in the video?',
        'video_choice_auto': 'Auto (up to {max})',
        'video_save_cancelled': '❌ Save cancelled',
        'video_pick_title': 'Choose video reference',
        'video_filter': 'Video (*.mp4 *.webm *.mov *.avi *.mkv)',
        'dance_name_title': 'Dance name', 'dance_name_prompt': 'Name for this dance?',
        'play_chars_group': 'Characters',
        'play_ref_label': 'Reference:', 'play_my_label': 'Me on camera:',
        'play_auto_big': 'Auto (largest)', 'play_auto_lead': '👑 Auto (lead)',
        'play_char_n': '#{n} · {name}', 'play_char_n_noname': 'Character {n}',
        'play_hint': 'ℹ️ "Auto (largest)" is quick. "Auto (lead)" is stable.',
        'play_choose': 'Choose a dance:', 'play_start': '▶  PLAY  DANCE',
        'play_stop': '■ Stop', 'play_playing': 'Playing: {name}{who}',
        'play_who_char': ' · #{n} {name}', 'play_who_auto_lead': ' · 👑 auto (lead)',
        'verdict_perfect': 'PERFECT!',
        'verdict_good': 'GOOD',
        'verdict_ok': 'OK',
        'verdict_miss': 'MISS',
        'dance_item': '{prefix} {name}\n{dur} · {frames} frames{suffix}',
        'dance_item_editor': '{prefix} {name}{suffix}',
        'dance_custom_names': '  ·  {names}',
        'dance_n_chars': '  ·  {k} pers.',
        'dance_names_plus': '{shown} +{rest}',
        'dance_audio_badge': '  ·  🎵',
        'stats_title': 'Dance finished!',
        'stats_character': 'Character: {who}',
        'stats_of': 'out of {max} points',
        'stats_cat_perfect': 'PERFECT', 'stats_cat_good': 'GOOD',
        'stats_cat_ok': 'OK', 'stats_cat_miss': 'MISS',
        'stats_desc_perfect': 'perfect', 'stats_desc_good': 'good',
        'stats_desc_ok': 'ok', 'stats_desc_miss': 'miss',
        'stats_frames': '{count} frames  ({pct:.0f}%)',
        'stats_avg': 'Average: {avg:.1f} / 100',
        'stats_combo': 'Max combo: {n} 🔥',
        'stats_total': 'Total: {n} frames', 'stats_done': 'Done',
        'names_title': 'Character names',
        'names_header': 'Detected characters: {n}',
        'names_hint': 'Set names (optional). Empty fields become "Character N".',
        'names_default': 'Character {n}',
        'dlg_error': 'Error', 'dlg_warning': 'Warning', 'dlg_info': 'Info',
        'btn_ok': 'OK', 'btn_cancel': 'Cancel',
        'err_no_poses_cam': 'MediaPipe does not see anyone.',
        'err_no_frames': 'Record a dance first.',
        'err_pick_dance_names': 'Select a dance in the list.',
        'err_only_one_char': 'This dance has only one character.',
        'err_save_video_copy': 'Could not copy video: {err}',
        'err_save_audio_copy': 'Could not copy audio: {err}',
        'err_delete_title': 'Delete',
        'msg_dance_saved': 'Dance saved:\n{path}',
        'msg_dance_saved_chars': '\n\nCharacters:\n{chars}',
        'msg_dance_saved_video': '\n\nVideo: {path}',
        'msg_dance_saved_audio': '\n\nAudio: {path}',
        'msg_video_saved': (
            'Dance "{name}" saved!\nPoses: {poses}\nCharacters: {chars}\n'
            'Duration: {dur:.1f}s'
        ),
        'msg_names_updated': 'Character names updated.',
        'msg_delete_confirm': 'Delete "{name}"?',
        'fullscreen_toggle_tip': 'Toggle fullscreen (F11)',
        'fullscreen_exit_btn': '✕ Exit fullscreen',
        'video_audio_group': '🎵 Audio (optional)',
        'video_audio_none': 'Not selected — will use video sound',
        'video_audio_pick': '🎵 Choose audio for this video',
        'video_audio_clear': '✕ Remove',
        'video_audio_hint': 'Use if the reference video has no sound or you want to replace it.',
        'video_audio_btn': '🎵 Audio',
        'video_audio_edit_title': 'Audio for dance',
        'video_audio_edit_prompt': 'Choose audio file (Remove — delete audio):',
        'video_audio_updated': 'Audio attached to dance.',
        'video_audio_removed': 'Audio removed from dance.',
        'audio_selected': '🎵 {name}',
    },
    'ru': {
        'nav_menu': 'Меню', 'nav_editor': '🎬 Редактор', 'nav_play': '🎮 Играть',
        'cam_group': 'Камера', 'cam_rescan': 'Обновить',
        'cam_rescan_tip': 'Пересканировать камеры',
        'cam_hint': 'Если камера не открылась — нажмите «Обновить».',
        'cam_opening': 'Открываю камеру #{n}…',
        'cam_status': 'Камера #{n}: {fps:.0f} FPS',
        'cam_status_people': 'Камера #{n}: {fps:.0f} FPS  ·  людей: {k}',
        'cam_error_not_found': 'Камера #{n} не открылась.',
        'cam_error_title': 'Камера', 'cam_not_found': 'Камеры не найдены',
        'cam_initializing': 'Инициализация камеры…',
        'welcome': 'Добро пожаловать!',
        'menu_info': (
            '🎬 <b>Редактор</b> — два способа создать танец:<br>'
            '&nbsp;&nbsp;• 🎥 <b>Запись на камеру</b> — станцуйте под музыку.<br>'
            '&nbsp;&nbsp;• 📹 <b>Видео-эталон</b> — загрузите mp4/webm. '
            'Выберите, сколько человек танцует (до {max}).<br><br>'
            '🏷 При сохранении можно задать имена персонажей.<br>'
            '👑 В игре есть <b>«Авто (ведущий)»</b>.<br><br>'
            '🖥 <b>F11</b> — полный экран, <b>Esc</b> — выход.<br><br>'
            '🏆 Итоговый балл: <b>0 – {score_max}</b>.'
        ),
        'lang_label': 'Язык:',
        'tab_cam': '🎥 Запись на камеру', 'tab_video': '📹 Видео-эталон',
        'music_group': '1. Музыка', 'music_none': 'Не выбрано',
        'music_pick': '📁 Выбрать аудио', 'music_pick_title': 'Выбрать аудио',
        'music_filter': 'Аудио (*.mp3 *.wav *.ogg *.m4a *.flac)',
        'name_group': '2. Название и видео', 'name_placeholder': 'Мой крутой танец',
        'record_video': '🎥 Записывать видео моего танца',
        'rec_group': '3. Запись танца',
        'rec_start': '● Начать', 'rec_stop': '■ Стоп',
        'rec_hint_cam': 'ℹ️ Если в кадре несколько человек, сохранятся все (до {max}).',
        'rec_frames_zero': 'Кадров: 0', 'rec_countdown': 'Отсчёт 3…',
        'rec_done': 'Готово. Кадров: {n}',
        'rec_done_people': 'Готово. Кадров: {n}  ·  макс. людей: {k}',
        'rec_live': 'Кадров: {n}  |  {t:.1f}s',
        'rec_live_people': 'Кадров: {n}  |  {t:.1f}s  ·  людей: {k}',
        'save_dance': '💾 Сохранить танец', 'saved_dances_group': 'Сохранённые танцы',
        'del_dance': '🗑 Удалить', 'edit_names': '🏷 Имена',
        'video_info': (
            'Загрузите видео с танцем (mp4, webm, mov, avi, mkv). '
            'MediaPipe отслеживает до {max} человек. Можно указать, сколько '
            'человек танцует, или использовать «Авто».<br><br>'
            'Во время игры видео воспроизводится с оверлеем — счёт поверх.'
        ),
        'video_pick': '📹 Выбрать видео-эталон',
        'video_processing_btn': '⏳ Обработка видео…',
        'video_processing_status': 'Распознаю позы…',
        'video_progress': 'Обработка: {cur} / {total} кадров ({pct}%)',
        'video_done': '✅ Готово! Поз: {n}  ·  персонажей: {k}',
        'video_open_error': 'Не удалось открыть видеофайл',
        'video_no_poses': 'В видео не обнаружено поз.',
        'video_processing_error': 'Ошибка обработки видео',
        'video_choice_title': 'Персонажи в видео',
        'video_choice_prompt': 'Сколько человек танцует в видео?',
        'video_choice_auto': 'Авто (до {max})',
        'video_save_cancelled': '❌ Сохранение отменено',
        'video_pick_title': 'Выбрать видео-эталон',
        'video_filter': 'Видео (*.mp4 *.webm *.mov *.avi *.mkv)',
        'dance_name_title': 'Название танца', 'dance_name_prompt': 'Как назвать танец?',
        'play_chars_group': 'Персонажи',
        'play_ref_label': 'Эталон:', 'play_my_label': 'Я на камере:',
        'play_auto_big': 'Авто (самый крупный)', 'play_auto_lead': '👑 Авто (ведущий)',
        'play_char_n': '#{n} · {name}', 'play_char_n_noname': 'Персонаж {n}',
        'play_hint': 'ℹ️ «Авто (крупный)» — быстро. «Авто (ведущий)» — стабильно.',
        'play_choose': 'Выберите танец:', 'play_start': '▶  ИГРАТЬ  ТАНЕЦ',
        'play_stop': '■ Остановить', 'play_playing': 'Играю: {name}{who}',
        'play_who_char': ' · #{n} {name}', 'play_who_auto_lead': ' · 👑 авто (ведущий)',
        'verdict_perfect': 'ИДЕАЛЬНО!',
        'verdict_good': 'ХОРОШО',
        'verdict_ok': 'НОРМ',
        'verdict_miss': 'МИМО',
        'dance_item': '{prefix} {name}\n{dur} · {frames} кадров{suffix}',
        'dance_item_editor': '{prefix} {name}{suffix}',
        'dance_custom_names': '  ·  {names}',
        'dance_n_chars': '  ·  {k} перс.',
        'dance_names_plus': '{shown} +{rest}',
        'dance_audio_badge': '  ·  🎵',
        'stats_title': 'Танец завершён!',
        'stats_character': 'Персонаж: {who}',
        'stats_of': 'из {max} очков',
        'stats_cat_perfect': 'PERFECT', 'stats_cat_good': 'GOOD',
        'stats_cat_ok': 'OK', 'stats_cat_miss': 'MISS',
        'stats_desc_perfect': 'идеально', 'stats_desc_good': 'хорошо',
        'stats_desc_ok': 'сойдёт', 'stats_desc_miss': 'мимо',
        'stats_frames': '{count} кадров  ({pct:.0f}%)',
        'stats_avg': 'Средний балл: {avg:.1f} / 100',
        'stats_combo': 'Макс. комбо: {n} 🔥',
        'stats_total': 'Всего проверено: {n} кадров', 'stats_done': 'Готово',
        'names_title': 'Имена персонажей',
        'names_header': 'Обнаружено персонажей: {n}',
        'names_hint': 'Задайте имена (необязательно).',
        'names_default': 'Персонаж {n}',
        'dlg_error': 'Ошибка', 'dlg_warning': 'Внимание', 'dlg_info': 'Информация',
        'btn_ok': 'OK', 'btn_cancel': 'Отмена',
        'err_no_poses_cam': 'MediaPipe не видит ни одного человека.',
        'err_no_frames': 'Сначала запишите танец.',
        'err_pick_dance_names': 'Выберите танец в списке.',
        'err_only_one_char': 'В этом танце только один персонаж.',
        'err_save_video_copy': 'Не удалось скопировать видео: {err}',
        'err_save_audio_copy': 'Не удалось скопировать аудио: {err}',
        'err_delete_title': 'Удалить',
        'msg_dance_saved': 'Танец сохранён:\n{path}',
        'msg_dance_saved_chars': '\n\nПерсонажи:\n{chars}',
        'msg_dance_saved_video': '\n\nВидео: {path}',
        'msg_dance_saved_audio': '\n\nАудио: {path}',
        'msg_video_saved': (
            'Танец «{name}» сохранён!\nПоз: {poses}\nПерсонажей: {chars}\n'
            'Длительность: {dur:.1f}с'
        ),
        'msg_names_updated': 'Имена персонажей обновлены.',
        'msg_delete_confirm': 'Удалить «{name}»?',
        'fullscreen_toggle_tip': 'Полноэкранный режим (F11)',
        'fullscreen_exit_btn': '✕ Выйти',
        'video_audio_group': '🎵 Аудио (опционально)',
        'video_audio_none': 'Не выбрано — будет звук самого видео',
        'video_audio_pick': '🎵 Выбрать аудио для этого видео',
        'video_audio_clear': '✕ Убрать',
        'video_audio_hint': 'Используйте, если у видео-эталона нет звука или нужно его заменить.',
        'video_audio_btn': '🎵 Аудио',
        'video_audio_edit_title': 'Аудио для танца',
        'video_audio_edit_prompt': 'Выберите аудиофайл (Убрать — удалить аудио):',
        'video_audio_updated': 'Аудио прикреплено к танцу.',
        'video_audio_removed': 'Аудио убрано из танца.',
        'audio_selected': '🎵 {name}',
    },
}

DEFAULT_LANG = 'en'
_current_lang = DEFAULT_LANG


def set_language(lang: str):
    global _current_lang
    _current_lang = lang if lang in STRINGS else DEFAULT_LANG


def current_language() -> str:
    return _current_lang


def tr(key: str, **kwargs) -> str:
    s = STRINGS.get(_current_lang, STRINGS[DEFAULT_LANG]).get(key)
    if s is None:
        s = STRINGS[DEFAULT_LANG].get(key, key)
    if kwargs:
        try:
            return s.format(**kwargs)
        except Exception:
            return s
    return s


# ============================================================
# Paths & constants
# ============================================================
def get_app_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.resolve()
    return Path(__file__).parent.resolve()


def get_bundle_dir() -> Path:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS).resolve()
    return get_app_dir()


APP_DIR = get_app_dir()
BUNDLE_DIR = get_bundle_dir()

DANCES_DIR = APP_DIR / "dances"
VIDEOS_DIR = APP_DIR / "recordings"
REFS_DIR = APP_DIR / "references"
MODELS_DIR = APP_DIR / "models"
for d in (DANCES_DIR, VIDEOS_DIR, REFS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

KEY_LM = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
POSE_CONN = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

MAX_POSES = 4
COL_WHITE = (255, 255, 255)
COL_TARGET = (255, 229, 0)
COL_GOOD = (136, 255, 0)
COL_WARN = (32, 176, 255)
COL_BAD = (139, 61, 255)
COL_REC = (68, 34, 255)
COL_OTHER = (110, 110, 110)

THRESH_PERFECT = 80
THRESH_GOOD    = 65
THRESH_OK      = 50

SIMILARITY_DIVISOR = 1.3

SCORE_MAX = 10000
LOOKAHEAD_SEC = 0.8
PREVIEW_SIZE = 180
VIDEO_TARGET_FPS = 15
REF_AUTO_LEAD = -2


# ============================================================
# File helpers
# ============================================================
def copy_to_refs(src_path: str, prefix: str = "asset") -> str | None:
    if not src_path:
        return None
    src = Path(src_path)
    if not src.exists():
        return None
    try:
        if src.resolve().parent == REFS_DIR.resolve():
            return str(src)
    except Exception:
        pass
    ext = src.suffix.lower() or ".bin"
    new_name = f"{prefix}_{uuid.uuid4().hex[:10]}{ext}"
    dst = REFS_DIR / new_name
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception as e:
        print(f"[copy_to_refs] failed: {e}")
        return None


# ============================================================
# Tracking utils
# ============================================================
def draw_skeleton_cv(frame, landmarks, color, line_w=3, alpha=1.0):
    if not landmarks:
        return
    h, w = frame.shape[:2]
    overlay = frame if alpha >= 1.0 else frame.copy()
    for a, b in POSE_CONN:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        p1, p2 = landmarks[a], landmarks[b]
        if len(p1) >= 4 and (p1[3] < 0.3 or p2[3] < 0.3):
            continue
        cv2.line(overlay,
                 (int(p1[0] * w), int(p1[1] * h)),
                 (int(p2[0] * w), int(p2[1] * h)),
                 color, line_w, cv2.LINE_AA)
    for i in KEY_LM:
        if i >= len(landmarks):
            continue
        p = landmarks[i]
        if len(p) >= 4 and p[3] < 0.3:
            continue
        cv2.circle(overlay, (int(p[0] * w), int(p[1] * h)),
                   6, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def extract_keypoints(landmarks):
    if landmarks is None or len(landmarks) < 29:
        return None
    lh, rh = landmarks[23], landmarks[24]
    ls, rs = landmarks[11], landmarks[12]
    hx, hy = (lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0
    sx, sy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    torso = ((sx - hx) ** 2 + (sy - hy) ** 2) ** 0.5
    if torso < 1e-6:
        torso = 0.001
    return [((landmarks[i][0] - hx) / torso,
             (landmarks[i][1] - hy) / torso) for i in KEY_LM]


def pose_distance(a, b):
    if a is None or b is None:
        return 999.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(a, b):
        dx, dy = x1 - x2, y1 - y2
        s += dx * dx + dy * dy
    return (s / len(a)) ** 0.5


def similarity_score(user, target):
    d = pose_distance(user, target)
    return 100.0 * float(np.exp(-d / SIMILARITY_DIVISOR))


def find_frame_at(frames, t):
    if not frames:
        return None
    lo, hi = 0, len(frames) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if frames[m]['t'] < t:
            lo = m + 1
        else:
            hi = m
    if lo > 0 and abs(frames[lo - 1]['t'] - t) < abs(frames[lo]['t'] - t):
        return frames[lo - 1]
    return frames[lo]


# ============================================================
# Multi-person helpers
# ============================================================
def get_poses_from_frame(frame):
    if 'poses' in frame:
        return frame['poses']
    if 'lm' in frame:
        return [frame['lm']]
    return []


def select_pose(poses, idx):
    if not poses:
        return None
    if idx == REF_AUTO_LEAD:
        return pick_lead_pose(poses)
    if 0 <= idx < len(poses):
        return poses[idx]
    return poses[0]


def dance_character_count(dance):
    if 'max_characters' in dance:
        return max(1, min(MAX_POSES, int(dance['max_characters'])))
    frames = dance.get('frames', [])
    max_n = 1
    for f in frames:
        if 'poses' in f:
            max_n = max(max_n, len(f['poses']))
        if max_n >= MAX_POSES:
            break
    return max(1, min(MAX_POSES, max_n))


def dance_character_names(dance) -> list:
    n = dance_character_count(dance)
    names = dance.get('character_names') or []
    result = []
    for i in range(n):
        if i < len(names) and names[i]:
            result.append(str(names[i]))
        else:
            result.append(tr('names_default', n=i + 1))
    return result


def pick_default_pose(poses):
    if not poses:
        return None
    best, best_area = None, -1.0
    for lm in poses:
        kp = extract_keypoints(lm)
        if not kp:
            continue
        xs = [p[0] for p in kp]
        ys = [p[1] for p in kp]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = lm
    return best if best is not None else poses[0]


def pose_lead_score(landmarks):
    if not landmarks or len(landmarks) < 29:
        return 0.0
    xs = [landmarks[i][0] for i in KEY_LM if i < len(landmarks)]
    ys = [landmarks[i][1] for i in KEY_LM if i < len(landmarks)]
    if not xs:
        return 0.0
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    size_score = min(1.0, (bw * bh) / 0.35)
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    center_score = max(0.0, 1.0 - dist * 2.2)
    vis_sum, vis_cnt = 0.0, 0
    for i in KEY_LM:
        if i < len(landmarks) and len(landmarks[i]) >= 4:
            vis_sum += landmarks[i][3]
            vis_cnt += 1
    vis_score = (vis_sum / vis_cnt) if vis_cnt else 0.0
    return 0.55 * size_score + 0.30 * center_score + 0.15 * vis_score


def pick_lead_pose(poses):
    if not poses:
        return None
    best, best_s = None, -1.0
    for lm in poses:
        s = pose_lead_score(lm)
        if s > best_s:
            best_s = s
            best = lm
    return best if best is not None else poses[0]


class LeadTracker:
    def __init__(self, switch_penalty=0.15):
        self.idx = None
        self.switch_penalty = switch_penalty

    def update(self, poses):
        if not poses:
            self.idx = None
            return None
        scores = [pose_lead_score(lm) for lm in poses]
        if self.idx is not None and 0 <= self.idx < len(poses):
            for i in range(len(scores)):
                if i != self.idx:
                    scores[i] -= self.switch_penalty
        best_i = int(np.argmax(scores))
        self.idx = best_i
        return best_i


# ============================================================
# Pose preview
# ============================================================
def render_pose_preview(landmarks, size=PREVIEW_SIZE, char_label=None) -> QPixmap:
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[:, :] = (40, 20, 20, 190)
    cv2.putText(img, "NEXT", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 230, 255), 1, cv2.LINE_AA)
    if char_label:
        cv2.putText(img, char_label, (size - 96, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 229, 0, 255), 1, cv2.LINE_AA)
    if landmarks and len(landmarks) >= 29:
        pts = [landmarks[i] for i in KEY_LM if i < len(landmarks)]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            bw = max(max(xs) - min(xs), 1e-3)
            bh = max(max(ys) - min(ys), 1e-3)
            pad_x, pad_y_top, pad_y_bot = 16, 34, 16
            avail_w = size - 2 * pad_x
            avail_h = size - pad_y_top - pad_y_bot
            scale = min(avail_w / bw, avail_h / bh)

            def to_px(p):
                x = (p[0] - cx) * scale + size / 2
                y = (p[1] - cy) * scale + (pad_y_top + avail_h / 2)
                return int(x), int(y)

            for a, b in POSE_CONN:
                if a >= len(landmarks) or b >= len(landmarks):
                    continue
                p1, p2 = landmarks[a], landmarks[b]
                if p1[3] < 0.3 or p2[3] < 0.3:
                    continue
                cv2.line(img, to_px(p1), to_px(p2),
                         (*COL_TARGET, 235), 3, cv2.LINE_AA)
            for i in KEY_LM:
                if i >= len(landmarks):
                    continue
                p = landmarks[i]
                if p[3] < 0.3:
                    continue
                cv2.circle(img, to_px(p), 5, (*COL_TARGET, 255), -1, cv2.LINE_AA)
    cv2.rectangle(img, (1, 1), (size - 2, size - 2),
                  (255, 229, 0, 120), 1, cv2.LINE_AA)
    img = np.ascontiguousarray(img)
    qimg = QImage(img.data, size, size, size * 4,
                  QImage.Format.Format_ARGB32).copy()
    return QPixmap.fromImage(qimg)


# ============================================================
# Model & cameras
# ============================================================
def ensure_pose_model() -> str:
    if POSE_MODEL_PATH.exists():
        return str(POSE_MODEL_PATH)
    bundled = BUNDLE_DIR / "models" / "pose_landmarker_lite.task"
    if bundled.exists() and bundled.resolve() != POSE_MODEL_PATH.resolve():
        try:
            shutil.copy2(bundled, POSE_MODEL_PATH)
            return str(POSE_MODEL_PATH)
        except Exception:
            return str(bundled)
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Could not fetch model: {e}")
    return str(POSE_MODEL_PATH)


def make_pose_landmarker(num_poses: int = MAX_POSES):
    model_path = ensure_pose_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=max(1, min(MAX_POSES, num_poses)),
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def camera_backends():
    if sys.platform == 'win32':
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform.startswith('linux'):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_camera(index: int):
    for backend in camera_backends():
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def enumerate_cameras(max_check: int = 6):
    qt_names = []
    try:
        for dev in QMediaDevices.videoInputs():
            try:
                name = dev.description() or "Camera"
            except Exception:
                name = "Camera"
            qt_names.append(name)
    except Exception:
        pass
    found = []
    for i in range(max_check):
        cap = open_camera(i)
        if cap is None:
            continue
        name = qt_names[i] if i < len(qt_names) else f"Camera {i}"
        found.append({'index': i, 'name': name})
        cap.release()
    return found


# ============================================================
# Dance storage
# ============================================================
def dances_list():
    items = []
    for p in sorted(DANCES_DIR.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                d = json.load(f)
            d['_path'] = str(p)
            items.append(d)
        except Exception as e:
            print(f"[dances] failed to read {p}: {e}")
    return items


def save_dance_dict(data: dict) -> Path:
    path = DANCES_DIR / f"{data['id']}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return path


# ============================================================
# Dialogs
# ============================================================
class CharacterNamesDialog(QDialog):
    def __init__(self, count: int, default_names=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('names_title'))
        self.setModal(True)
        self.setMinimumWidth(380)
        self.setStyleSheet("QDialog{background:#16162e;} QLabel{color:#e8e8ff;}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)

        title = QLabel(tr('names_header', n=count))
        title.setStyleSheet("font-size:15px;font-weight:800;color:#fff;")
        layout.addWidget(title)

        hint = QLabel(tr('names_hint'))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8a8ac0;font-size:11px;")
        layout.addWidget(hint)

        self.edits = []
        for i in range(count):
            row = QHBoxLayout()
            lbl = QLabel(f"#{i + 1}:")
            lbl.setStyleSheet("color:#8a8ac0;font-size:12px;")
            lbl.setFixedWidth(30)
            ed = QLineEdit()
            ed.setPlaceholderText(tr('names_default', n=i + 1))
            if default_names and i < len(default_names) and default_names[i]:
                ed.setText(str(default_names[i]))
            ed.setStyleSheet("""
                QLineEdit{background:#0f0f24;color:#e8e8ff;
                    border:1px solid #2a2a52;border-radius:8px;
                    padding:8px 10px;font-size:13px;}
                QLineEdit:focus{border-color:#00e5ff;}
            """)
            row.addWidget(lbl)
            row.addWidget(ed, 1)
            layout.addLayout(row)
            self.edits.append(ed)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText(tr('btn_ok'))
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('btn_cancel'))
        btns.setStyleSheet("""
            QPushButton{background:#2a2a52;color:#fff;
                padding:8px 16px;border-radius:6px;border:none;
                min-width:80px;font-size:12px;font-weight:700;}
            QPushButton:hover{background:#3a3a72;}
        """)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.names = []

    def accept(self):
        self.names = []
        for i, ed in enumerate(self.edits):
            txt = ed.text().strip()
            self.names.append(txt if txt else tr('names_default', n=i + 1))
        super().accept()


class StatsDialog(QDialog):
    def __init__(self, stats: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr('stats_title'))
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setStyleSheet("QDialog{background:#16162e;} QLabel{color:#e8e8ff;}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 26, 30, 26)
        layout.setSpacing(12)

        title = QLabel(f"🏁 {tr('stats_title')}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:22px;font-weight:900;color:#fff;")
        layout.addWidget(title)

        if stats.get('character_label'):
            who = QLabel(tr('stats_character', who=stats['character_label']))
            who.setAlignment(Qt.AlignmentFlag.AlignCenter)
            who.setStyleSheet("color:#00e5ff;font-size:13px;font-weight:700;")
            layout.addWidget(who)

        big = QLabel(f"{stats['score']}")
        big.setAlignment(Qt.AlignmentFlag.AlignCenter)
        big.setStyleSheet(
            "font-size:64px;font-weight:900;color:#00e5ff;"
            "padding:4px 0 0 0;letter-spacing:2px;"
        )
        layout.addWidget(big)

        sub = QLabel(tr('stats_of', max=SCORE_MAX))
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color:#8a8ac0;font-size:12px;margin-top:-14px;")
        layout.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#2a2a52;border:none;margin:8px 0;")
        layout.addWidget(sep)

        total = max(1, stats['total'])
        rows = [
            (tr('stats_cat_perfect'), stats['perfect'], "#00ff88", tr('stats_desc_perfect')),
            (tr('stats_cat_good'), stats['good'], "#00e5ff", tr('stats_desc_good')),
            (tr('stats_cat_ok'), stats['ok'], "#ffb020", tr('stats_desc_ok')),
            (tr('stats_cat_miss'), stats['miss'], "#ff3d8b", tr('stats_desc_miss')),
        ]
        for label, count, color, desc in rows:
            row = QHBoxLayout()
            name = QLabel(f"{label}  ·  {desc}")
            name.setStyleSheet(f"color:{color};font-weight:800;font-size:13px;")
            pct = count / total * 100
            val = QLabel(tr('stats_frames', count=count, pct=pct))
            val.setStyleSheet("color:#b8b8e0;font-size:12px;")
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(name)
            row.addStretch()
            row.addWidget(val)
            layout.addLayout(row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background:#2a2a52;border:none;margin:8px 0;")
        layout.addWidget(sep2)

        avg = QLabel(tr('stats_avg', avg=stats['avg']))
        avg.setStyleSheet("color:#e8e8ff;font-size:14px;font-weight:700;")
        avg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(avg)

        combo = QLabel(tr('stats_combo', n=stats['max_combo']))
        combo.setStyleSheet("color:#e8e8ff;font-size:14px;font-weight:700;")
        combo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(combo)

        tot = QLabel(tr('stats_total', n=stats['total']))
        tot.setStyleSheet("color:#8a8ac0;font-size:12px;")
        tot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tot)

        btn = QPushButton(tr('stats_done'))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 #00e5ff, stop:1 #0066cc);
                color:#fff;border:none;padding:12px;
                border-radius:8px;font-size:14px;font-weight:800;
                margin-top:10px;}
            QPushButton:hover{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 #0066cc, stop:1 #00e5ff);}
        """)
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ============================================================
# Camera worker (with low-latency patches)
# ============================================================
class CameraWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    fps_update = pyqtSignal(float)
    camera_error = pyqtSignal(str)

    def __init__(self, cam_index: int = 0):
        super().__init__()
        self.cam_index = cam_index
        self._running = False
        self._recorder = None
        self._pending_record_path = None
        self._recording_video = False

    def start_video_recording(self, path):
        self._pending_record_path = str(path)

    def stop_video_recording(self):
        self._recording_video = False
        if self._recorder is not None:
            self._recorder.release()
            self._recorder = None

    def run(self):
        self._running = True
        cap = open_camera(self.cam_index)
        if cap is None:
            self.camera_error.emit(tr('cam_error_not_found', n=self.cam_index))
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Минимальный буфер — всегда свежий кадр
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            landmarker = make_pose_landmarker(num_poses=MAX_POSES)
        except Exception as e:
            self.camera_error.emit(str(e))
            cap.release()
            return

        fps_t0, fps_count = time.time(), 0
        frame_ts_ms = int(time.time() * 1000)

        while self._running:
            # Drain — пропускаем накопившиеся кадры, берём свежий
            for _ in range(3):
                cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
                if not ret:
                    self.msleep(20)
                    continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if self._pending_record_path is not None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._recorder = cv2.VideoWriter(
                    self._pending_record_path, fourcc, 20.0, (w, h)
                )
                self._recording_video = True
                self._pending_record_path = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Реальный wall-clock timestamp
            frame_ts_ms = int(time.time() * 1000)
            try:
                result = landmarker.detect_for_video(mp_image, frame_ts_ms)
            except Exception:
                result = None

            poses = []
            if result is not None and result.pose_landmarks:
                for pl in result.pose_landmarks:
                    poses.append([
                        (p.x, p.y, p.z, getattr(p, 'visibility', 1.0))
                        for p in pl
                    ])

            if self._recording_video and self._recorder is not None:
                rec = frame.copy()
                for lm in poses:
                    draw_skeleton_cv(rec, lm, COL_WHITE, 3, 1.0)
                self._recorder.write(rec)

            self.frame_ready.emit(frame, poses)

            fps_count += 1
            if fps_count >= 15:
                t = time.time()
                self.fps_update.emit(15.0 / max(1e-3, t - fps_t0))
                fps_t0, fps_count = t, 0

        cap.release()
        self.stop_video_recording()
        try:
            landmarker.close()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait(2500)


# ============================================================
# Video processor
# ============================================================
class VideoProcessor(QThread):
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, src_path: str, num_poses_hint: int = MAX_POSES):
        super().__init__()
        self.src_path = src_path
        self.num_poses_hint = max(1, min(MAX_POSES, int(num_poses_hint)))

    def run(self):
        try:
            cap = cv2.VideoCapture(self.src_path)
            if not cap.isOpened():
                self.failed.emit(tr('video_open_error'))
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            duration = total / fps if fps > 0 else 0.0
            sample_step = max(1, int(round(fps / VIDEO_TARGET_FPS)))

            landmarker = make_pose_landmarker(num_poses=self.num_poses_hint)

            frames = []
            max_chars = 1
            idx = 0
            last_reported = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % sample_step == 0:
                    t = idx / fps
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    try:
                        result = landmarker.detect_for_video(mp_image, int(t * 1000))
                    except Exception:
                        result = None
                    if result is not None and result.pose_landmarks:
                        poses = [
                            [(p.x, p.y, p.z, getattr(p, 'visibility', 1.0))
                             for p in pl]
                            for pl in result.pose_landmarks
                        ]
                        if poses:
                            max_chars = max(max_chars, len(poses))
                            frames.append({'t': t, 'poses': poses})
                idx += 1
                if total > 0 and idx % 15 == 0 and idx != last_reported:
                    self.progress.emit(idx, total)
                    last_reported = idx

            cap.release()
            try:
                landmarker.close()
            except Exception:
                pass

            if not frames:
                self.failed.emit(tr('video_no_poses'))
                return

            self.finished_ok.emit({
                'frames': frames,
                'duration': duration,
                'max_characters': max_chars,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


# ============================================================
# Main window
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💃 DanceMaster")
        self.resize(1320, 800)

        self.is_fullscreen = False
        self.cam_index = 0
        self.camera: CameraWorker | None = None

        self.current_mode = 'menu'
        self.current_frame = None
        self.all_poses = []
        self.current_landmarks = None
        self.current_keypoints = None
        self.prev_my_kp = None

        self.my_char_idx = -1
        self.ref_char_idx = 0
        self.ref_lead_tracker = LeadTracker()
        self.my_lead_tracker = LeadTracker()

        self.ed_recording = False
        self.ed_frames = []
        self.ed_name = ""
        self.ed_audio_path = None
        self.ed_video_path = None
        self.ed_record_video = True
        self.ed_last_capture = 0.0
        self.ed_clock = QElapsedTimer()
        self.ed_countdown_until = 0.0
        self.ed_max_chars = 1

        self.video_processor: VideoProcessor | None = None
        self.video_proc_src = None
        self.video_proc_name = ""
        self.ref_audio_path = None

        self.pl_active = False
        self.pl_dance = None
        self.pl_score_sum = 0.0
        self.pl_score_max = 0.0
        self.pl_last_tick = 0.0
        self.pl_target = None
        self.pl_clock = QElapsedTimer()
        self.pl_last_score = 0.0
        self.pl_is_video = False
        self.pl_stats = self._empty_stats()
        self.pl_stats_last_update = 0.0
        self.pl_preview_last_update = 0.0
        self.pl_finishing = False
        self.verdict_current = None

        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.85)

        self.video_player = QMediaPlayer()
        self.video_audio_output = QAudioOutput()
        self.video_player.setAudioOutput(self.video_audio_output)
        self.video_audio_output.setVolume(0.85)

        self.video_sink = QVideoSink()
        self.video_player.setVideoSink(self.video_sink)
        self.video_sink.videoFrameChanged.connect(self.on_video_frame)
        self.current_video_image: QImage | None = None

        self.settings = QSettings("DanceMaster", "DanceMaster")
        saved_lang = self.settings.value("language", DEFAULT_LANG)
        set_language(str(saved_lang))

        self._build_ui()
        self.retranslate()
        self._setup_shortcuts()

        QTimer.singleShot(100, self.refresh_cameras)
        QTimer.singleShot(200, lambda: self._start_camera(self.cam_index))

        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update_display)
        # 16 мс ≈ 60 FPS — минимизируем задержку отрисовки
        self.render_timer.start(16)

    @staticmethod
    def _empty_stats():
        return {
            'total': 0, 'perfect': 0, 'good': 0, 'ok': 0, 'miss': 0,
            'sum': 0.0, 'max_combo': 0, 'cur_combo': 0,
        }

    # ---------------------------------------------------------
    # Video frame from reference video
    # ---------------------------------------------------------
    def on_video_frame(self, frame: QVideoFrame):
        if not frame.isValid():
            return
        img = frame.toImage()
        if img.isNull():
            return
        self.current_video_image = img.convertToFormat(
            QImage.Format.Format_RGB888
        )

    # ---------------------------------------------------------
    # FULLSCREEN
    # ---------------------------------------------------------
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Esc"), self, activated=self._on_escape)

    def _on_escape(self):
        if self.is_fullscreen:
            self.exit_fullscreen()

    def toggle_fullscreen(self):
        if self.is_fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self):
        if self.is_fullscreen:
            return
        self.is_fullscreen = True
        self.panel.hide()
        self.fullscreen_exit_btn.setVisible(True)
        self.fullscreen_exit_btn.raise_()
        self.showFullScreen()
        QTimer.singleShot(0, self._force_rerender)
        QTimer.singleShot(60, self._force_rerender)
        QTimer.singleShot(200, self._force_rerender)

    def exit_fullscreen(self):
        if not self.is_fullscreen:
            return
        self.is_fullscreen = False
        self.panel.show()
        self.fullscreen_exit_btn.setVisible(False)
        self.showNormal()
        QTimer.singleShot(0, self._force_rerender)
        QTimer.singleShot(60, self._force_rerender)

    def _force_rerender(self):
        cw = self.centralWidget()
        if cw is not None and cw.layout() is not None:
            cw.layout().activate()
        if hasattr(self, 'stage_container'):
            self.stage_container.updateGeometry()
            self.stage_container.update()
        if hasattr(self, 'video_label'):
            self.video_label.updateGeometry()
            self.video_label.update()
        self.update_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.update_display)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == event.Type.WindowStateChange:
            QTimer.singleShot(0, self._force_rerender)

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        stage = QFrame()
        stage.setStyleSheet("background:#000;")
        stage.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)

        self.stage_container = QWidget()
        self.stage_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        grid = QGridLayout(self.stage_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background:#000;color:#666;font-size:16px;"
        )
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.video_label.setMinimumSize(1, 1)
        grid.addWidget(self.video_label, 0, 0)

        self.pose_preview_label = QLabel()
        self.pose_preview_label.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self.pose_preview_label.setStyleSheet("background:transparent;")
        self.pose_preview_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.pose_preview_label.setVisible(False)
        grid.addWidget(
            self.pose_preview_label, 0, 0,
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
        )

        self.verdict_label = QLabel()
        self.verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.verdict_label.setVisible(False)
        grid.addWidget(
            self.verdict_label, 0, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        self.verdict_hide_timer = QTimer(self)
        self.verdict_hide_timer.setSingleShot(True)
        self.verdict_hide_timer.timeout.connect(self._hide_verdict)

        self.fullscreen_exit_btn = QPushButton(tr('fullscreen_exit_btn'))
        self.fullscreen_exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fullscreen_exit_btn.setStyleSheet("""
            QPushButton{background:rgba(0,0,0,0.65);color:#fff;
                border:1px solid rgba(255,255,255,0.25);
                padding:8px 14px;border-radius:8px;
                font-size:13px;font-weight:700;}
            QPushButton:hover{background:rgba(255,61,139,0.85);}
        """)
        self.fullscreen_exit_btn.clicked.connect(self.exit_fullscreen)
        self.fullscreen_exit_btn.setVisible(False)
        grid.addWidget(
            self.fullscreen_exit_btn, 0, 0,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
        )

        stage_layout.addWidget(self.stage_container, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet("""
            QProgressBar{background:#111;border:none;}
            QProgressBar::chunk{background:qlineargradient(
                x1:0,y1:0,x2:1,y2:0,stop:0 #00e5ff, stop:1 #ff3d8b);}
        """)
        stage_layout.addWidget(self.progress)
        root.addWidget(stage, 1)

        self.panel = QFrame()
        self.panel.setFixedWidth(420)
        self.panel.setStyleSheet("background:#16162e;")
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)

        # ---- header (paddings reduced to move icons up) ----
        header = QFrame()
        header.setStyleSheet("background:#0f0f24;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 8, 12, 4)

        self.title_label = QLabel("💃 DanceMaster")
        self.title_label.setStyleSheet(
            "color:#fff;font-size:17px;font-weight:900;"
        )
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setFixedSize(28, 28)
        self.fullscreen_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fullscreen_btn.setToolTip(tr('fullscreen_toggle_tip'))
        self.fullscreen_btn.setStyleSheet("""
            QPushButton{background:#1a1a3a;color:#e8e8ff;
                border:1px solid #2a2a52;border-radius:6px;
                font-size:15px;font-weight:700;}
            QPushButton:hover{background:#241238;border-color:#00e5ff;}
        """)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        header_layout.addWidget(self.fullscreen_btn)
        panel_layout.addWidget(header)

        # ---- nav (paddings reduced) ----
        nav = QFrame()
        nav.setStyleSheet("background:#0f0f24;")
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(12, 0, 12, 6)
        nav_layout.setSpacing(6)

        self.btn_menu = QPushButton("")
        self.btn_editor = QPushButton("")
        self.btn_play = QPushButton("")
        for b, m in [(self.btn_menu, 'menu'),
                     (self.btn_editor, 'editor'),
                     (self.btn_play, 'play')]:
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton{background:transparent;color:#8a8ac0;
                    border:1px solid transparent;padding:6px 10px;
                    border-radius:8px;font-size:12px;font-weight:700;}
                QPushButton:hover{color:#fff;background:#1a1a3a;}
                QPushButton:checked{color:#fff;border:none;
                    background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #ff3d8b, stop:1 #8a2be2);}
            """)
            b.clicked.connect(lambda _, mm=m: self.set_mode(mm))
            nav_layout.addWidget(b)
        self.btn_menu.setChecked(True)
        panel_layout.addWidget(nav)

        self.stack = QStackedWidget()
        panel_layout.addWidget(self.stack, 1)

        self.page_menu = QWidget()
        self.page_editor = QWidget()
        self.page_play = QWidget()
        self.stack.addWidget(self.page_menu)
        self.stack.addWidget(self.page_editor)
        self.stack.addWidget(self.page_play)

        self._build_menu_page()
        self._build_editor_page()
        self._build_play_page()

        self.status = QStatusBar()
        self.status.setStyleSheet(
            "background:#0f0f24;color:#8a8ac0;border-top:1px solid #2a2a52;"
        )
        self.status.showMessage(tr('cam_initializing'))
        panel_layout.addWidget(self.status)

        root.addWidget(self.panel)

    def _build_menu_page(self):
        layout = QVBoxLayout(self.page_menu)
        layout.setContentsMargins(20, 14, 20, 20)
        layout.setSpacing(12)

        self.cam_group = QGroupBox()
        self.cam_group.setStyleSheet(self._group_css())
        l_cam = QVBoxLayout(self.cam_group)

        row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setStyleSheet(self._combo_css())
        self.cam_combo.currentIndexChanged.connect(self.on_camera_selected)
        row.addWidget(self.cam_combo, 1)

        self.btn_rescan = QPushButton("🔄")
        self.btn_rescan.setFixedWidth(44)
        self.btn_rescan.setStyleSheet(self._btn_css())
        self.btn_rescan.clicked.connect(self.refresh_cameras)
        row.addWidget(self.btn_rescan)
        l_cam.addLayout(row)

        self.cam_hint = QLabel()
        self.cam_hint.setWordWrap(True)
        self.cam_hint.setStyleSheet(
            "color:#8a8ac0;font-size:11px;margin-top:4px;"
        )
        l_cam.addWidget(self.cam_hint)
        layout.addWidget(self.cam_group)

        self.lang_group = QGroupBox()
        self.lang_group.setStyleSheet(self._group_css())
        l_lang = QHBoxLayout(self.lang_group)
        self.lang_label = QLabel()
        self.lang_label.setStyleSheet("color:#8a8ac0;font-size:12px;")
        l_lang.addWidget(self.lang_label)

        self.lang_combo = QComboBox()
        self.lang_combo.setStyleSheet(self._combo_css())
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Русский", "ru")
        self.lang_combo.setCurrentIndex(0 if current_language() == 'en' else 1)
        self.lang_combo.currentIndexChanged.connect(self.on_lang_changed)
        l_lang.addWidget(self.lang_combo, 1)
        layout.addWidget(self.lang_group)

        self.welcome_label = QLabel()
        self.welcome_label.setStyleSheet(
            "color:#fff;font-size:18px;font-weight:800;"
        )
        layout.addWidget(self.welcome_label)

        self.menu_info_label = QLabel()
        self.menu_info_label.setWordWrap(True)
        self.menu_info_label.setStyleSheet(
            "color:#b8b8e0;font-size:13px;line-height:1.6;"
        )
        layout.addWidget(self.menu_info_label)
        layout.addStretch()

    def _build_editor_page(self):
        layout = QVBoxLayout(self.page_editor)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        self.editor_tabs = QTabWidget()
        self.editor_tabs.setStyleSheet("""
            QTabWidget::pane{border:1px solid #2a2a52;border-radius:8px;
                background:#0f0f24;padding:8px;top:-1px;}
            QTabBar::tab{background:#1a1a3a;color:#8a8ac0;
                padding:8px 12px;border-top-left-radius:6px;
                border-top-right-radius:6px;margin-right:2px;
                font-size:11px;font-weight:700;}
            QTabBar::tab:selected{background:#241238;color:#fff;}
            QTabBar::tab:hover{background:#2a2a52;color:#fff;}
        """)

        tab_cam = QWidget()
        self._build_editor_camera_tab(tab_cam)
        self.editor_tabs.addTab(tab_cam, "")

        tab_vid = QWidget()
        self._build_editor_video_tab(tab_vid)
        self.editor_tabs.addTab(tab_vid, "")

        layout.addWidget(self.editor_tabs)

        self.saved_group = QGroupBox()
        self.saved_group.setStyleSheet(self._group_css())
        l_list = QVBoxLayout(self.saved_group)

        self.dances_list_editor = QListWidget()
        self.dances_list_editor.setStyleSheet(self._list_css())
        self.dances_list_editor.setFixedHeight(140)
        l_list.addWidget(self.dances_list_editor)

        btn_row = QHBoxLayout()
        self.btn_del_dance = QPushButton()
        self.btn_del_dance.setStyleSheet(self._btn_css("#ff4466", "#8a1a3a"))
        self.btn_del_dance.clicked.connect(self.delete_selected_dance)
        self.btn_edit_names = QPushButton()
        self.btn_edit_names.setStyleSheet(self._btn_css())
        self.btn_edit_names.clicked.connect(self.edit_selected_dance_names)
        self.btn_edit_audio = QPushButton()
        self.btn_edit_audio.setStyleSheet(self._btn_css())
        self.btn_edit_audio.clicked.connect(self.edit_selected_dance_audio)
        btn_row.addWidget(self.btn_del_dance)
        btn_row.addWidget(self.btn_edit_names)
        btn_row.addWidget(self.btn_edit_audio)
        l_list.addLayout(btn_row)
        layout.addWidget(self.saved_group)
        layout.addStretch()

    def _build_editor_camera_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.music_group = QGroupBox()
        self.music_group.setStyleSheet(self._group_css())
        l1 = QVBoxLayout(self.music_group)
        self.lbl_audio = QLabel()
        self.lbl_audio.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.lbl_audio.setWordWrap(True)
        l1.addWidget(self.lbl_audio)
        self.btn_pick_audio = QPushButton()
        self.btn_pick_audio.setStyleSheet(self._btn_css())
        self.btn_pick_audio.clicked.connect(self.pick_audio)
        l1.addWidget(self.btn_pick_audio)
        layout.addWidget(self.music_group)

        self.name_group = QGroupBox()
        self.name_group.setStyleSheet(self._group_css())
        l2 = QVBoxLayout(self.name_group)
        self.inp_name = QLineEdit()
        self.inp_name.setStyleSheet(self._input_css())
        self.inp_name.textChanged.connect(lambda s: setattr(self, 'ed_name', s))
        l2.addWidget(self.inp_name)

        self.chk_video = QCheckBox()
        self.chk_video.setChecked(True)
        self.chk_video.setStyleSheet("color:#e8e8ff;font-size:13px;")
        self.chk_video.stateChanged.connect(self._on_chk_video)
        l2.addWidget(self.chk_video)
        layout.addWidget(self.name_group)

        self.rec_group = QGroupBox()
        self.rec_group.setStyleSheet(self._group_css())
        l3 = QVBoxLayout(self.rec_group)

        row = QHBoxLayout()
        self.btn_rec_start = QPushButton()
        self.btn_rec_start.setStyleSheet(self._btn_css("#ff3d8b", "#8a2be2"))
        self.btn_rec_start.clicked.connect(self.start_recording)
        self.btn_rec_stop = QPushButton()
        self.btn_rec_stop.setStyleSheet(self._btn_css("#ff4466", "#8a1a3a"))
        self.btn_rec_stop.clicked.connect(self.stop_recording)
        self.btn_rec_stop.setEnabled(False)
        row.addWidget(self.btn_rec_start)
        row.addWidget(self.btn_rec_stop)
        l3.addLayout(row)

        self.rec_hint = QLabel()
        self.rec_hint.setWordWrap(True)
        self.rec_hint.setStyleSheet("color:#8a8ac0;font-size:11px;margin-top:2px;")
        l3.addWidget(self.rec_hint)

        self.lbl_rec_status = QLabel()
        self.lbl_rec_status.setStyleSheet("color:#8a8ac0;font-size:12px;")
        l3.addWidget(self.lbl_rec_status)

        self.btn_save = QPushButton()
        self.btn_save.setStyleSheet(self._btn_css("#00e5ff", "#0066cc"))
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_camera_dance)
        l3.addWidget(self.btn_save)
        layout.addWidget(self.rec_group)
        layout.addStretch()

    def _build_editor_video_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self.video_info_label = QLabel()
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet(
            "color:#b8b8e0;font-size:12px;line-height:1.5;"
        )
        layout.addWidget(self.video_info_label)

        self.ref_audio_group = QGroupBox()
        self.ref_audio_group.setStyleSheet(self._group_css())
        l_au = QVBoxLayout(self.ref_audio_group)

        self.lbl_ref_audio = QLabel()
        self.lbl_ref_audio.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.lbl_ref_audio.setWordWrap(True)
        l_au.addWidget(self.lbl_ref_audio)

        row_au = QHBoxLayout()
        self.btn_pick_ref_audio = QPushButton()
        self.btn_pick_ref_audio.setStyleSheet(self._btn_css())
        self.btn_pick_ref_audio.clicked.connect(self.pick_ref_audio)
        row_au.addWidget(self.btn_pick_ref_audio, 1)

        self.btn_clear_ref_audio = QPushButton()
        self.btn_clear_ref_audio.setStyleSheet(
            self._btn_css("#ff4466", "#8a1a3a")
        )
        self.btn_clear_ref_audio.clicked.connect(self.clear_ref_audio)
        self.btn_clear_ref_audio.setEnabled(False)
        row_au.addWidget(self.btn_clear_ref_audio)
        l_au.addLayout(row_au)

        self.ref_audio_hint = QLabel()
        self.ref_audio_hint.setWordWrap(True)
        self.ref_audio_hint.setStyleSheet(
            "color:#8a8ac0;font-size:11px;margin-top:2px;"
        )
        l_au.addWidget(self.ref_audio_hint)
        layout.addWidget(self.ref_audio_group)

        self.btn_video_upload = QPushButton()
        self.btn_video_upload.setStyleSheet(self._btn_css("#00e5ff", "#0066cc"))
        self.btn_video_upload.setMinimumHeight(48)
        self.btn_video_upload.clicked.connect(self.pick_reference_video)
        layout.addWidget(self.btn_video_upload)

        self.lbl_video_status = QLabel("")
        self.lbl_video_status.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.lbl_video_status.setWordWrap(True)
        layout.addWidget(self.lbl_video_status)

        self.video_prog = QProgressBar()
        self.video_prog.setRange(0, 100)
        self.video_prog.setValue(0)
        self.video_prog.setTextVisible(True)
        self.video_prog.setStyleSheet("""
            QProgressBar{background:#0f0f24;color:#e8e8ff;
                border:1px solid #2a2a52;border-radius:6px;
                text-align:center;height:22px;font-size:11px;}
            QProgressBar::chunk{background:qlineargradient(
                x1:0,y1:0,x2:1,y2:0,stop:0 #00e5ff, stop:1 #0066cc);
                border-radius:5px;}
        """)
        layout.addWidget(self.video_prog)
        layout.addStretch()

    def _build_play_page(self):
        layout = QVBoxLayout(self.page_play)
        layout.setContentsMargins(20, 10, 20, 16)
        layout.setSpacing(10)

        # ============ Characters group ============
        self.char_group = QGroupBox()
        self.char_group.setStyleSheet(self._group_css())
        l_char = QVBoxLayout(self.char_group)
        l_char.setSpacing(6)

        row1 = QHBoxLayout()
        self.play_ref_label = QLabel()
        self.play_ref_label.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.play_ref_label.setFixedWidth(90)
        self.ref_combo = QComboBox()
        self.ref_combo.setStyleSheet(self._combo_css())
        self.ref_combo.addItem("", 0)
        self.ref_combo.setEnabled(False)
        self.ref_combo.currentIndexChanged.connect(self.on_ref_char_changed)
        row1.addWidget(self.play_ref_label)
        row1.addWidget(self.ref_combo, 1)
        l_char.addLayout(row1)

        row2 = QHBoxLayout()
        self.play_my_label = QLabel()
        self.play_my_label.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.play_my_label.setFixedWidth(90)
        self.my_combo = QComboBox()
        self.my_combo.setStyleSheet(self._combo_css())
        self.my_combo.addItem("", -1)
        self.my_combo.addItem("", -2)
        for i in range(MAX_POSES):
            self.my_combo.addItem("", i)
        self.my_combo.currentIndexChanged.connect(self.on_my_char_changed)
        row2.addWidget(self.play_my_label)
        row2.addWidget(self.my_combo, 1)
        l_char.addLayout(row2)

        self.play_hint = QLabel()
        self.play_hint.setWordWrap(True)
        self.play_hint.setStyleSheet("color:#8a8ac0;font-size:11px;")
        l_char.addWidget(self.play_hint)
        layout.addWidget(self.char_group)

        # ============ PLAY DANCE — moved to top ============
        self.btn_play_sel = QPushButton()
        self.btn_play_sel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play_sel.setFixedHeight(58)
        self.btn_play_sel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_play_sel.setStyleSheet("""
            QPushButton{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #00e5ff, stop:1 #0088ff);
                color:#ffffff;
                border:2px solid #7cf0ff;
                padding:14px 18px;
                border-radius:12px;
                font-size:16px;
                font-weight:900;
                letter-spacing:2px;
            }
            QPushButton:hover{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #4df0ff, stop:1 #00a8ff);
                border-color:#ffffff;
            }
            QPushButton:pressed{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0088ff, stop:1 #0055cc);
            }
            QPushButton:disabled{
                background:#1e1e3a;
                color:#555580;
                border-color:#2a2a52;
            }
        """)
        self.btn_play_sel.clicked.connect(self.play_selected)
        layout.addWidget(self.btn_play_sel)

        self.btn_play_stop = QPushButton()
        self.btn_play_stop.setStyleSheet(self._btn_css("#ff4466", "#8a1a3a"))
        self.btn_play_stop.setFixedHeight(40)
        self.btn_play_stop.setEnabled(False)
        self.btn_play_stop.clicked.connect(self.stop_play)
        layout.addWidget(self.btn_play_stop)

        # ============ Dance list ============
        self.play_choose_label = QLabel()
        self.play_choose_label.setStyleSheet(
            "color:#8a8ac0;font-size:12px;font-weight:700;margin-top:6px;"
        )
        layout.addWidget(self.play_choose_label)

        self.dance_list = QListWidget()
        self.dance_list.setStyleSheet(self._list_css())
        self.dance_list.itemDoubleClicked.connect(self.play_selected)
        self.dance_list.currentItemChanged.connect(self.on_dance_selection_changed)
        layout.addWidget(self.dance_list, 1)

        self.lbl_play_status = QLabel("")
        self.lbl_play_status.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.lbl_play_status.setWordWrap(True)
        layout.addWidget(self.lbl_play_status)

        self.refresh_dance_lists()

    # --- CSS helpers ---
    @staticmethod
    def _group_css():
        return """
            QGroupBox{color:#a0a0d0;font-size:12px;font-weight:700;
                border:1px solid #2a2a52;border-radius:10px;
                margin-top:10px;padding:14px 10px 10px 10px;
                letter-spacing:1px;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 6px;}
        """

    @staticmethod
    def _btn_css(c1="#2a2a52", c2="#1a1a3a"):
        return f"""
            QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {c1}, stop:1 {c2});
                color:#fff;border:none;padding:10px 14px;
                border-radius:8px;font-size:13px;font-weight:700;}}
            QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 {c2}, stop:1 {c1});}}
            QPushButton:disabled{{background:#222244;color:#555580;}}
        """

    @staticmethod
    def _input_css():
        return """
            QLineEdit{background:#0f0f24;color:#e8e8ff;
                border:1px solid #2a2a52;border-radius:8px;
                padding:10px 12px;font-size:13px;}
            QLineEdit:focus{border-color:#00e5ff;}
        """

    @staticmethod
    def _list_css():
        return """
            QListWidget{background:#0f0f24;color:#e8e8ff;
                border:1px solid #2a2a52;border-radius:8px;
                padding:6px;font-size:12px;}
            QListWidget::item{padding:8px;border-radius:6px;}
            QListWidget::item:hover{background:#1a1a3a;}
            QListWidget::item:selected{background:#241238;color:#fff;}
        """

    @staticmethod
    def _combo_css():
        return """
            QComboBox{background:#0f0f24;color:#e8e8ff;
                border:1px solid #2a2a52;border-radius:8px;
                padding:6px 10px;font-size:12px;}
            QComboBox:hover{border-color:#00e5ff;}
            QComboBox::drop-down{border:none;width:20px;}
            QComboBox QAbstractItemView{background:#16162e;color:#e8e8ff;
                selection-background-color:#241238;
                border:1px solid #2a2a52;}
        """

    # --- language ---
    def on_lang_changed(self, combo_index: int):
        lang = self.lang_combo.itemData(combo_index)
        if not lang:
            return
        set_language(str(lang))
        self.settings.setValue("language", current_language())
        self.retranslate()
        self.refresh_dance_lists()

    def retranslate(self):
        self.btn_menu.setText(tr('nav_menu'))
        self.btn_editor.setText(tr('nav_editor'))
        self.btn_play.setText(tr('nav_play'))

        self.cam_group.setTitle(tr('cam_group'))
        self.btn_rescan.setToolTip(tr('cam_rescan_tip'))
        self.cam_hint.setText(tr('cam_hint'))
        self.lang_group.setTitle(tr('lang_label').rstrip(':'))
        self.welcome_label.setText(tr('welcome'))
        self.menu_info_label.setText(
            tr('menu_info', max=MAX_POSES, score_max=SCORE_MAX)
        )

        self.editor_tabs.setTabText(0, tr('tab_cam'))
        self.editor_tabs.setTabText(1, tr('tab_video'))

        self.music_group.setTitle(tr('music_group'))
        self.btn_pick_audio.setText(tr('music_pick'))
        self.name_group.setTitle(tr('name_group'))
        self.inp_name.setPlaceholderText(tr('name_placeholder'))
        self.chk_video.setText(tr('record_video'))
        self.rec_group.setTitle(tr('rec_group'))
        self.btn_rec_start.setText(tr('rec_start'))
        self.btn_rec_stop.setText(tr('rec_stop'))
        self.rec_hint.setText(tr('rec_hint_cam', max=MAX_POSES))
        if not self.ed_recording:
            self.lbl_rec_status.setText(tr('rec_frames_zero'))
        self.btn_save.setText(tr('save_dance'))
        if not self.ed_audio_path:
            self.lbl_audio.setText(tr('music_none'))

        self.video_info_label.setText(tr('video_info', max=MAX_POSES))
        if not (self.video_processor is not None
                and self.video_processor.isRunning()):
            self.btn_video_upload.setText(tr('video_pick'))

        self.ref_audio_group.setTitle(tr('video_audio_group'))
        self.btn_pick_ref_audio.setText(tr('video_audio_pick'))
        self.btn_clear_ref_audio.setText(tr('video_audio_clear'))
        self.ref_audio_hint.setText(tr('video_audio_hint'))
        self.btn_edit_audio.setText(tr('video_audio_btn'))
        if not self.ref_audio_path:
            self.lbl_ref_audio.setText(tr('video_audio_none'))

        self.saved_group.setTitle(tr('saved_dances_group'))
        self.btn_del_dance.setText(tr('del_dance'))
        self.btn_edit_names.setText(tr('edit_names'))

        self.char_group.setTitle(tr('play_chars_group'))
        self.play_ref_label.setText(tr('play_ref_label'))
        self.play_my_label.setText(tr('play_my_label'))
        self.play_hint.setText(tr('play_hint'))
        self.play_choose_label.setText(tr('play_choose'))
        self.btn_play_sel.setText(tr('play_start'))
        self.btn_play_stop.setText(tr('play_stop'))

        self._rebuild_my_combo()

        self.fullscreen_btn.setToolTip(tr('fullscreen_toggle_tip'))
        self.fullscreen_exit_btn.setText(tr('fullscreen_exit_btn'))

        if not self.pl_active and not self.ed_recording:
            self.status.showMessage(tr('cam_initializing'))

    def _rebuild_my_combo(self):
        current = self.my_combo.currentData()
        self.my_combo.blockSignals(True)
        self.my_combo.clear()
        self.my_combo.addItem(tr('play_auto_big'), -1)
        self.my_combo.addItem(tr('play_auto_lead'), -2)
        for i in range(MAX_POSES):
            self.my_combo.addItem(tr('play_char_n_noname', n=i + 1), i)
        idx = self.my_combo.findData(current)
        if idx >= 0:
            self.my_combo.setCurrentIndex(idx)
        self.my_combo.blockSignals(False)

    # --- camera ---
    def refresh_cameras(self):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        cams = enumerate_cameras()
        if not cams:
            self.cam_combo.addItem(tr('cam_not_found'), -1)
            self.cam_combo.setEnabled(False)
        else:
            for c in cams:
                self.cam_combo.addItem(f"[{c['index']}] {c['name']}", c['index'])
            self.cam_combo.setEnabled(True)
            idx = self.cam_combo.findData(self.cam_index)
            if idx >= 0:
                self.cam_combo.setCurrentIndex(idx)
        self.cam_combo.blockSignals(False)

    def on_camera_selected(self, combo_index: int):
        data = self.cam_combo.itemData(combo_index)
        if data is None or data < 0:
            return
        if data == self.cam_index and self.camera is not None:
            return
        self._start_camera(int(data))

    def _start_camera(self, index: int):
        self._stop_camera()
        self.cam_index = index
        self.camera = CameraWorker(cam_index=index)
        self.camera.frame_ready.connect(self.on_frame)
        self.camera.fps_update.connect(self.on_fps)
        self.camera.camera_error.connect(self.on_camera_error)
        self.camera.start()
        self.status.showMessage(tr('cam_opening', n=index))

    def _stop_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

    # --- modes ---
    def set_mode(self, mode):
        if self.ed_recording and mode != 'editor':
            self.stop_recording()
        if self.pl_active and mode != 'play':
            self.stop_play()
        self.current_mode = mode
        self.btn_menu.setChecked(mode == 'menu')
        self.btn_editor.setChecked(mode == 'editor')
        self.btn_play.setChecked(mode == 'play')
        self.stack.setCurrentIndex({'menu': 0, 'editor': 1, 'play': 2}[mode])
        if mode in ('play', 'editor'):
            self.refresh_dance_lists()

    def _on_chk_video(self, state):
        self.ed_record_video = (state == Qt.CheckState.Checked.value)

    # --- character selection ---
    def on_ref_char_changed(self, combo_index: int):
        data = self.ref_combo.itemData(combo_index)
        if data is None:
            return
        self.ref_char_idx = int(data)
        self.ref_lead_tracker = LeadTracker()

    def on_my_char_changed(self, combo_index: int):
        data = self.my_combo.itemData(combo_index)
        if data is None:
            return
        self.my_char_idx = int(data)
        self.prev_my_kp = None
        self.my_lead_tracker = LeadTracker()

    def on_dance_selection_changed(self, current, previous):
        if current is None:
            return
        dance = current.data(Qt.ItemDataRole.UserRole)
        names = dance_character_names(dance)
        n = len(names)

        self.ref_combo.blockSignals(True)
        self.ref_combo.clear()
        for i, nm in enumerate(names):
            self.ref_combo.addItem(tr('play_char_n', n=i + 1, name=nm), i)
        if n > 1:
            self.ref_combo.addItem(tr('play_auto_lead'), REF_AUTO_LEAD)
        self.ref_combo.setCurrentIndex(0)
        self.ref_combo.setEnabled(n > 1)
        self.ref_combo.blockSignals(False)
        self.ref_char_idx = 0

        prev_data = self.my_combo.currentData()
        self.my_combo.blockSignals(True)
        self.my_combo.clear()
        self.my_combo.addItem(tr('play_auto_big'), -1)
        self.my_combo.addItem(tr('play_auto_lead'), -2)
        for i, nm in enumerate(names):
            self.my_combo.addItem(tr('play_char_n', n=i + 1, name=nm), i)
        idx = self.my_combo.findData(prev_data)
        if idx >= 0:
            self.my_combo.setCurrentIndex(idx)
        self.my_combo.blockSignals(False)

    # --- frame input (low latency: update_display called immediately) ---
    def on_frame(self, frame, poses):
        self.current_frame = frame
        self.all_poses = poses if poses else []
        self.track_user_pose(self.all_poses)

        if self.current_mode == 'editor' and self.ed_recording:
            self.capture_editor_frame()
        if self.current_mode == 'play' and self.pl_active:
            self.score_play_frame()
            self.update_pose_preview()
            # Немедленная отрисовка — не ждём render_timer
            self.update_display()

    def track_user_pose(self, poses):
        if not poses:
            self.current_landmarks = None
            self.current_keypoints = None
            return

        chosen = None
        if self.my_char_idx == REF_AUTO_LEAD:
            idx = self.my_lead_tracker.update(poses)
            if idx is not None and 0 <= idx < len(poses):
                chosen = poses[idx]
        elif 0 <= self.my_char_idx < len(poses):
            chosen = poses[self.my_char_idx]
        else:
            if self.prev_my_kp is None:
                chosen = pick_default_pose(poses)
            else:
                best_d = float('inf')
                for lm in poses:
                    kp = extract_keypoints(lm)
                    if not kp:
                        continue
                    d = pose_distance(kp, self.prev_my_kp)
                    if d < best_d:
                        best_d = d
                        chosen = lm

        self.current_landmarks = chosen
        self.current_keypoints = extract_keypoints(chosen) if chosen else None
        if self.current_keypoints is not None:
            self.prev_my_kp = self.current_keypoints

    def on_fps(self, fps):
        if not self.pl_active and not self.ed_recording:
            n = len(self.all_poses)
            if n > 1:
                self.status.showMessage(
                    tr('cam_status_people', n=self.cam_index, fps=fps, k=n)
                )
            else:
                self.status.showMessage(
                    tr('cam_status', n=self.cam_index, fps=fps)
                )

    def on_camera_error(self, msg):
        self.status.showMessage(msg)
        QMessageBox.warning(self, tr('cam_error_title'), msg)

    # --- pose preview ---
    def update_pose_preview(self):
        if not self.pl_active or not self.pl_dance:
            return
        now = time.time()
        if now - self.pl_preview_last_update < 0.1:
            return
        self.pl_preview_last_update = now

        t_next = self.play_clock() + LOOKAHEAD_SEC
        frame = find_frame_at(self.pl_dance.get('frames', []), t_next)
        if not frame:
            return
        poses = get_poses_from_frame(frame)

        if self.ref_char_idx == REF_AUTO_LEAD:
            idx = self.ref_lead_tracker.update(poses) or 0
            lm = poses[idx] if 0 <= idx < len(poses) else None
            label = "👑 auto"
        else:
            lm = select_pose(poses, self.ref_char_idx)
            label = None
            if len(poses) > 1:
                label = f"#{self.ref_char_idx + 1}"

        pix = render_pose_preview(lm, PREVIEW_SIZE, char_label=label)
        self.pose_preview_label.setPixmap(pix)

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------
    def _draw_play_overlays(self, painter: QPainter, w: int, h: int):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        score = self.play_score_display()
        if score >= THRESH_GOOD * SCORE_MAX / 100:
            sc_color = QColor("#00ff88")
        elif score >= THRESH_OK * SCORE_MAX / 100:
            sc_color = QColor("#ffb020")
        else:
            sc_color = QColor("#ff3d8b")

        text = str(score)
        f = QFont("Segoe UI", 34, QFont.Weight.Black)
        painter.setFont(f)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(text)
        painter.setPen(QColor(0, 0, 0, 180))
        painter.drawText(w - tw - 28, 62, text)
        painter.setPen(sc_color)
        painter.drawText(w - tw - 25, 60, text)

        combo = self.pl_stats['cur_combo']
        if combo > 1:
            f2 = QFont("Segoe UI", 22, QFont.Weight.Black)
            painter.setFont(f2)
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(21, h - 21, f"x{combo}")
            painter.setPen(QColor("#00ff88"))
            painter.drawText(20, h - 20, f"x{combo}")

    def update_display(self):
        # ---------- VIDEO REFERENCE MODE ----------
        if self.current_mode == 'play' and self.pl_active and self.pl_is_video:
            if self.current_video_image is None or self.current_video_image.isNull():
                return
            pix = QPixmap.fromImage(self.current_video_image)
            scaled = pix.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            if scaled.isNull():
                return
            painter = QPainter(scaled)
            self._draw_play_overlays(painter, scaled.width(), scaled.height())
            painter.end()
            self.video_label.setPixmap(scaled)
            return

        # ---------- CAMERA MODE ----------
        if self.current_frame is None:
            return

        display = self.current_frame.copy()
        h, w = display.shape[:2]

        if self.current_mode == 'play' and self.pl_active and self.pl_target:
            draw_skeleton_cv(display, self.pl_target, COL_TARGET, 3, 0.55)

        if self.all_poses:
            for lm in self.all_poses:
                is_me = (lm is self.current_landmarks)
                if is_me:
                    if self.current_mode == 'play' and self.pl_active:
                        s = self.pl_last_score
                        color = (COL_GOOD if s >= THRESH_GOOD else
                                 COL_WARN if s >= THRESH_OK else COL_BAD)
                    elif self.ed_recording:
                        color = COL_REC
                    else:
                        color = COL_WHITE
                    draw_skeleton_cv(display, lm, color, 3, 1.0)
                else:
                    draw_skeleton_cv(display, lm, COL_OTHER, 2, 0.35)

        if self.ed_recording:
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(display, (30, 30), 12, COL_REC, -1, cv2.LINE_AA)
            cv2.putText(display, "REC", (50, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if self.ed_countdown_until > 0:
                remaining = self.ed_countdown_until - time.time()
                if remaining > 0:
                    n = int(remaining) + 1
                    cv2.putText(display, str(n),
                                (w // 2 - 45, h // 2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 5.0,
                                (229, 229, 0), 8, cv2.LINE_AA)
            t = self.editor_clock()
            cv2.putText(display, f"{t:.1f}s", (w - 130, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if len(self.all_poses) > 1:
                cv2.putText(display, f"people: {len(self.all_poses)}",
                            (w - 220, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (180, 180, 220), 2, cv2.LINE_AA)

        if (self.current_mode == 'play' and self.pl_active
                and not self.pl_is_video):
            score = self.play_score_display()
            color = (COL_GOOD if score >= THRESH_GOOD * SCORE_MAX / 100 else
                     COL_WARN if score >= THRESH_OK * SCORE_MAX / 100 else COL_BAD)
            text = str(score)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
            cv2.putText(display, text, (w - tw - 25, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            if self.pl_stats['cur_combo'] > 1:
                cv2.putText(display, f"x{self.pl_stats['cur_combo']}",
                            (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            COL_GOOD, 2, cv2.LINE_AA)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        hh, ww, ch = rgb.shape
        img = QImage(rgb.data, ww, hh, ch * ww, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img)
        scaled = pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    # --- editor: camera recording ---
    def pick_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr('music_pick_title'), "", tr('music_filter')
        )
        if path:
            self.ed_audio_path = path
            self.lbl_audio.setText(tr('audio_selected', name=Path(path).name))
            self.lbl_audio.setStyleSheet("color:#00ff88;font-size:12px;")

    def pick_ref_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr('music_pick_title'), "", tr('music_filter')
        )
        if not path:
            return
        self.ref_audio_path = path
        self.lbl_ref_audio.setText(tr('audio_selected', name=Path(path).name))
        self.lbl_ref_audio.setStyleSheet("color:#00ff88;font-size:12px;")
        self.btn_clear_ref_audio.setEnabled(True)

    def clear_ref_audio(self):
        self.ref_audio_path = None
        self.lbl_ref_audio.setText(tr('video_audio_none'))
        self.lbl_ref_audio.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.btn_clear_ref_audio.setEnabled(False)

    def start_recording(self):
        if self.ed_recording:
            return
        if not self.all_poses:
            QMessageBox.warning(self, tr('dlg_warning'), tr('err_no_poses_cam'))
            return

        self.ed_frames = []
        self.ed_max_chars = 1
        self.ed_last_capture = 0.0
        self.ed_video_path = None
        self.ed_countdown_until = time.time() + 3
        self.lbl_rec_status.setText(tr('rec_countdown'))

        if self.ed_audio_path:
            self.audio_player.setSource(QUrl.fromLocalFile(self.ed_audio_path))
            self.audio_player.setPosition(0)

        QTimer.singleShot(3000, self._really_start_recording)

    def _really_start_recording(self):
        self.ed_countdown_until = 0
        self.ed_recording = True
        self.ed_clock.restart()
        self.ed_last_capture = 0.0

        if self.ed_audio_path:
            self.audio_player.play()

        if self.ed_record_video:
            fname = f"dance_{int(time.time())}.mp4"
            self.ed_video_path = str(VIDEOS_DIR / fname)
            if self.camera is not None:
                self.camera.start_video_recording(self.ed_video_path)

        self.btn_rec_start.setEnabled(False)
        self.btn_rec_stop.setEnabled(True)
        self.lbl_rec_status.setText(tr('rec_frames_zero'))

    def editor_clock(self):
        if (self.ed_audio_path and
                self.audio_player.playbackState() ==
                QMediaPlayer.PlaybackState.PlayingState):
            return self.audio_player.position() / 1000.0
        return self.ed_clock.elapsed() / 1000.0

    def capture_editor_frame(self):
        now = time.time()
        if now - self.ed_last_capture < 0.1:
            return
        self.ed_last_capture = now
        if not self.all_poses:
            return
        t = self.editor_clock()
        poses_data = []
        for lm in self.all_poses:
            poses_data.append([(p[0], p[1], p[2], p[3]) for p in lm])
        self.ed_max_chars = max(self.ed_max_chars, len(poses_data))
        self.ed_frames.append({'t': t, 'poses': poses_data})

        if len(poses_data) > 1:
            self.lbl_rec_status.setText(
                tr('rec_live_people', n=len(self.ed_frames), t=t, k=len(poses_data))
            )
        else:
            self.lbl_rec_status.setText(
                tr('rec_live', n=len(self.ed_frames), t=t)
            )

    def stop_recording(self):
        if not self.ed_recording:
            return
        self.ed_recording = False
        self.audio_player.stop()
        if self.camera is not None:
            self.camera.stop_video_recording()
        self.btn_rec_start.setEnabled(True)
        self.btn_rec_stop.setEnabled(False)
        self.btn_save.setEnabled(len(self.ed_frames) > 0)
        if self.ed_max_chars > 1:
            self.lbl_rec_status.setText(
                tr('rec_done_people', n=len(self.ed_frames), k=self.ed_max_chars)
            )
        else:
            self.lbl_rec_status.setText(
                tr('rec_done', n=len(self.ed_frames))
            )

    def save_camera_dance(self):
        if not self.ed_frames:
            QMessageBox.warning(self, tr('dlg_warning'), tr('err_no_frames'))
            return

        names = None
        if self.ed_max_chars > 1:
            dlg = CharacterNamesDialog(self.ed_max_chars, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            names = dlg.names

        saved_audio_path = None
        if self.ed_audio_path:
            copied = copy_to_refs(self.ed_audio_path, prefix="audio")
            saved_audio_path = copied if copied else self.ed_audio_path

        name = self.ed_name.strip() or f"Dance {time.strftime('%d.%m %H:%M')}"
        dance_id = uuid.uuid4().hex[:12]
        duration = self.ed_frames[-1]['t'] if self.ed_frames else 0.0
        data = {
            'id': dance_id, 'name': name, 'type': 'camera',
            'duration': duration, 'audio_path': saved_audio_path,
            'video_path': None, 'created_at': time.time(),
            'max_characters': self.ed_max_chars, 'frames': self.ed_frames,
        }
        if names:
            data['character_names'] = names
        path = save_dance_dict(data)

        msg = tr('msg_dance_saved', path=path.name)
        if names:
            chars = "\n".join(f"  #{i+1}: {nm}" for i, nm in enumerate(names))
            msg += tr('msg_dance_saved_chars', chars=chars)
        if saved_audio_path:
            msg += tr('msg_dance_saved_audio', path=saved_audio_path)
        if self.ed_video_path:
            msg += tr('msg_dance_saved_video', path=self.ed_video_path)
        QMessageBox.information(self, tr('dlg_info'), msg)

        self.ed_frames = []
        self.ed_name = ""
        self.ed_max_chars = 1
        self.inp_name.setText("")
        self.ed_audio_path = None
        self.ed_video_path = None
        self.lbl_audio.setText(tr('music_none'))
        self.lbl_audio.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.btn_save.setEnabled(False)
        self.lbl_rec_status.setText(tr('rec_frames_zero'))
        self.refresh_dance_lists()

    # --- editor: video reference ---
    def pick_reference_video(self):
        src, _ = QFileDialog.getOpenFileName(
            self, tr('video_pick_title'), "", tr('video_filter')
        )
        if not src:
            return

        name, ok = QInputDialog.getText(
            self, tr('dance_name_title'), tr('dance_name_prompt'),
            text=Path(src).stem
        )
        if not ok or not name.strip():
            return

        opts = [tr('video_choice_auto', max=MAX_POSES), "1", "2", "3", "4"]
        choice, ok = QInputDialog.getItem(
            self, tr('video_choice_title'), tr('video_choice_prompt'),
            opts, 0, False
        )
        if not ok:
            return
        if choice == opts[0]:
            num_poses = MAX_POSES
        else:
            try:
                num_poses = int(choice)
            except ValueError:
                num_poses = MAX_POSES

        self.video_proc_src = src
        self.video_proc_name = name.strip()

        self.video_processor = VideoProcessor(src, num_poses_hint=num_poses)
        self.video_processor.progress.connect(self.on_video_progress)
        self.video_processor.finished_ok.connect(self.on_video_done)
        self.video_processor.failed.connect(self.on_video_failed)
        self.video_processor.start()

        self.btn_video_upload.setEnabled(False)
        self.btn_video_upload.setText(tr('video_processing_btn'))
        self.video_prog.setValue(0)
        self.lbl_video_status.setText(tr('video_processing_status'))

    def on_video_progress(self, cur, total):
        if total > 0:
            pct = safe_pct_0_100(cur / total * 100)
            self.video_prog.setValue(pct)
            self.lbl_video_status.setText(
                tr('video_progress', cur=cur, total=total, pct=pct)
            )

    def on_video_done(self, result):
        self.btn_video_upload.setEnabled(True)
        self.btn_video_upload.setText(tr('video_pick'))
        self.video_prog.setValue(100)
        max_chars = result.get('max_characters', 1)
        self.lbl_video_status.setText(
            tr('video_done', n=len(result['frames']), k=max_chars)
        )

        names = None
        if max_chars > 1:
            dlg = CharacterNamesDialog(max_chars, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                self.lbl_video_status.setText(tr('video_save_cancelled'))
                return
            names = dlg.names

        dance_id = uuid.uuid4().hex[:12]
        ext = Path(self.video_proc_src).suffix.lower() or ".mp4"
        ref_path = REFS_DIR / f"{dance_id}{ext}"
        try:
            shutil.copy2(self.video_proc_src, ref_path)
        except Exception as e:
            QMessageBox.critical(self, tr('dlg_error'),
                                 tr('err_save_video_copy', err=str(e)))
            return

        saved_audio_path = None
        if self.ref_audio_path:
            copied = copy_to_refs(self.ref_audio_path, prefix="audio")
            saved_audio_path = copied if copied else self.ref_audio_path

        data = {
            'id': dance_id, 'name': self.video_proc_name, 'type': 'video',
            'duration': result['duration'],
            'audio_path': saved_audio_path,
            'video_path': str(ref_path), 'created_at': time.time(),
            'max_characters': max_chars, 'frames': result['frames'],
        }
        if names:
            data['character_names'] = names
        save_dance_dict(data)

        info = tr('msg_video_saved',
                  name=self.video_proc_name,
                  poses=len(result['frames']),
                  chars=max_chars,
                  dur=result['duration'])
        if names:
            info += "\n\n" + "\n".join(
                f"  #{i+1}: {nm}" for i, nm in enumerate(names)
            )
        if saved_audio_path:
            info += tr('msg_dance_saved_audio', path=saved_audio_path)
        QMessageBox.information(self, tr('dlg_info'), info)

        self.ref_audio_path = None
        self.lbl_ref_audio.setText(tr('video_audio_none'))
        self.lbl_ref_audio.setStyleSheet("color:#8a8ac0;font-size:12px;")
        self.btn_clear_ref_audio.setEnabled(False)

        self.refresh_dance_lists()

    def on_video_failed(self, msg):
        self.btn_video_upload.setEnabled(True)
        self.btn_video_upload.setText(tr('video_pick'))
        self.video_prog.setValue(0)
        self.lbl_video_status.setText(f"❌ {msg}")
        QMessageBox.critical(self, tr('video_processing_error'), msg)

    # --- dance list ---
    def refresh_dance_lists(self):
        items = dances_list()

        self.dance_list.clear()
        for d in items:
            dur = d.get('duration', 0)
            m, s = int(dur // 60), int(dur % 60)
            prefix = "📹" if d.get('type') == 'video' else "🎥"
            n_chars = dance_character_count(d)
            chars_suffix = ""
            if n_chars > 1:
                custom = d.get('character_names') or []
                has_custom = any(x and x.strip() for x in custom)
                if has_custom:
                    names = dance_character_names(d)
                    shown = ", ".join(names[:2])
                    if len(names) > 2:
                        shown = tr('dance_names_plus',
                                   shown=shown, rest=len(names) - 2)
                    chars_suffix = tr('dance_custom_names', names=shown)
                else:
                    chars_suffix = tr('dance_n_chars', k=n_chars)
            if d.get('audio_path') and Path(d['audio_path']).exists():
                chars_suffix += tr('dance_audio_badge')
            text = tr('dance_item',
                      prefix=prefix, name=d['name'],
                      dur=f"{m}:{s:02d}",
                      frames=len(d.get('frames', [])),
                      suffix=chars_suffix)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, d)
            self.dance_list.addItem(item)

        self.dances_list_editor.clear()
        for d in items:
            prefix = "📹" if d.get('type') == 'video' else "🎥"
            n_chars = dance_character_count(d)
            chars_suffix = ""
            custom = d.get('character_names') or []
            if n_chars > 1 and any(x and x.strip() for x in custom):
                names = dance_character_names(d)
                chars_suffix = tr('dance_custom_names', names=", ".join(names))
            elif n_chars > 1:
                chars_suffix = tr('dance_n_chars', k=n_chars)
            if d.get('audio_path') and Path(d['audio_path']).exists():
                chars_suffix += tr('dance_audio_badge')
            text = tr('dance_item_editor',
                      prefix=prefix, name=d['name'], suffix=chars_suffix)
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, d)
            self.dances_list_editor.addItem(item)

        if self.dance_list.count() > 0:
            self.on_dance_selection_changed(
                self.dance_list.currentItem(), None
            )

    def delete_selected_dance(self):
        item = self.dances_list_editor.currentItem()
        if not item:
            return
        dance = item.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(
            self, tr('err_delete_title'),
            tr('msg_delete_confirm', name=dance['name'])
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            Path(dance['_path']).unlink(missing_ok=True)
            if dance.get('video_path') and Path(dance['video_path']).exists():
                Path(dance['video_path']).unlink(missing_ok=True)
            ap = dance.get('audio_path')
            if ap:
                p = Path(ap)
                try:
                    if p.exists() and p.resolve().parent == REFS_DIR.resolve():
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"[dances] delete error: {e}")
        self.refresh_dance_lists()

    def edit_selected_dance_names(self):
        item = self.dances_list_editor.currentItem()
        if not item:
            QMessageBox.information(self, tr('dlg_info'),
                                    tr('err_pick_dance_names'))
            return
        dance = item.data(Qt.ItemDataRole.UserRole)
        n = dance_character_count(dance)
        if n <= 1:
            QMessageBox.information(self, tr('dlg_info'),
                                    tr('err_only_one_char'))
            return

        current = dance.get('character_names') or []
        dlg = CharacterNamesDialog(n, default_names=current, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        dance['character_names'] = dlg.names
        to_save = {k: v for k, v in dance.items() if not k.startswith('_')}
        save_dance_dict(to_save)
        QMessageBox.information(self, tr('dlg_info'), tr('msg_names_updated'))
        self.refresh_dance_lists()

    def edit_selected_dance_audio(self):
        item = self.dances_list_editor.currentItem()
        if not item:
            QMessageBox.information(self, tr('dlg_info'),
                                    tr('err_pick_dance_names'))
            return
        dance = item.data(Qt.ItemDataRole.UserRole)

        has_audio = bool(dance.get('audio_path')
                         and Path(dance['audio_path']).exists())

        if has_audio:
            box = QMessageBox(self)
            box.setWindowTitle(tr('video_audio_edit_title'))
            box.setText(tr('video_audio_edit_prompt'))
            b_pick = box.addButton(tr('music_pick'),
                                   QMessageBox.ButtonRole.AcceptRole)
            b_clear = box.addButton(tr('video_audio_clear'),
                                    QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(tr('btn_cancel'),
                          QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == b_pick:
                path, _ = QFileDialog.getOpenFileName(
                    self, tr('music_pick_title'), "", tr('music_filter')
                )
                if not path:
                    return
                copied = copy_to_refs(path, prefix="audio") or path
                old = dance.get('audio_path')
                if old and old != copied:
                    try:
                        op = Path(old)
                        if op.exists() and op.resolve().parent == REFS_DIR.resolve():
                            op.unlink(missing_ok=True)
                    except Exception:
                        pass
                dance['audio_path'] = copied
                to_save = {k: v for k, v in dance.items()
                           if not k.startswith('_')}
                save_dance_dict(to_save)
                QMessageBox.information(self, tr('dlg_info'),
                                        tr('video_audio_updated'))
            elif clicked == b_clear:
                old = dance.get('audio_path')
                if old:
                    try:
                        op = Path(old)
                        if op.exists() and op.resolve().parent == REFS_DIR.resolve():
                            op.unlink(missing_ok=True)
                    except Exception:
                        pass
                dance['audio_path'] = None
                to_save = {k: v for k, v in dance.items()
                           if not k.startswith('_')}
                save_dance_dict(to_save)
                QMessageBox.information(self, tr('dlg_info'),
                                        tr('video_audio_removed'))
            else:
                return
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, tr('music_pick_title'), "", tr('music_filter')
            )
            if not path:
                return
            copied = copy_to_refs(path, prefix="audio") or path
            dance['audio_path'] = copied
            to_save = {k: v for k, v in dance.items()
                       if not k.startswith('_')}
            save_dance_dict(to_save)
            QMessageBox.information(self, tr('dlg_info'),
                                    tr('video_audio_updated'))

        self.refresh_dance_lists()

    # ---------------------------------------------------------
    # POSE VERDICT
    # ---------------------------------------------------------
    def _show_verdict(self, category: str):
        colors = {
            'perfect': '#00ff88',
            'good':    '#00e5ff',
            'ok':      '#ffb020',
            'miss':    '#ff3d8b',
        }
        texts = {
            'perfect': tr('verdict_perfect'),
            'good':    tr('verdict_good'),
            'ok':      tr('verdict_ok'),
            'miss':    tr('verdict_miss'),
        }
        c = colors.get(category, '#ffffff')
        t = texts.get(category, '')
        self.verdict_label.setText(t)
        self.verdict_label.setStyleSheet(f"""
            QLabel{{
                color:{c};
                background:rgba(0,0,0,0.55);
                font-size:60px;
                font-weight:900;
                padding:8px 32px;
                border:3px solid {c};
                border-radius:18px;
            }}
        """)
        self.verdict_label.setVisible(True)
        self.verdict_label.raise_()
        self.verdict_hide_timer.start(800)

    def _hide_verdict(self):
        self.verdict_label.setVisible(False)

    def _reset_verdict(self):
        self.verdict_current = None
        self.verdict_label.setVisible(False)
        self.verdict_hide_timer.stop()

    # ---------------------------------------------------------
    # PLAYBACK FINISH DETECTION
    # ---------------------------------------------------------
    def _is_playback_finished(self) -> bool:
        EOM = QMediaPlayer.MediaStatus.EndOfMedia
        INVALID = QMediaPlayer.MediaStatus.InvalidMedia
        if self.pl_is_video:
            has_ext_audio = bool(
                self.pl_dance
                and self.pl_dance.get('audio_path')
                and Path(self.pl_dance['audio_path']).exists()
            )
            if has_ext_audio:
                st = self.audio_player.mediaStatus()
                return st in (EOM, INVALID)
            st = self.video_player.mediaStatus()
            return st in (EOM, INVALID)
        if (self.pl_dance
                and self.pl_dance.get('audio_path')
                and Path(self.pl_dance['audio_path']).exists()):
            st = self.audio_player.mediaStatus()
            return st in (EOM, INVALID)
        return False

    def play_selected(self):
        item = self.dance_list.currentItem()
        if not item:
            if self.dance_list.count() > 0:
                self.dance_list.setCurrentRow(0)
                item = self.dance_list.currentItem()
        if not item:
            QMessageBox.information(self, tr('dlg_info'),
                                    tr('err_pick_dance_names'))
            return
        dance = item.data(Qt.ItemDataRole.UserRole)
        self.start_play(dance)

    # --- play ---
    def start_play(self, dance):
        if self.pl_active:
            self.stop_play()

        self.ref_char_idx = int(self.ref_combo.currentData() or 0)
        my_data = self.my_combo.currentData()
        self.my_char_idx = int(my_data) if my_data is not None else -1
        self.prev_my_kp = None
        self.ref_lead_tracker = LeadTracker()
        self.my_lead_tracker = LeadTracker()

        self.pl_dance = dance
        self.pl_active = True
        self.pl_score_sum = 0.0
        self.pl_score_max = 0.0
        self.pl_last_tick = time.time()
        self.pl_target = None
        self.pl_last_score = 0.0
        self.pl_stats = self._empty_stats()
        self.pl_stats_last_update = 0.0
        self.pl_preview_last_update = 0.0
        self.pl_finishing = False
        self._reset_verdict()
        self.current_video_image = None

        video_path = dance.get('video_path')
        audio_path = dance.get('audio_path')
        has_audio = bool(audio_path and Path(audio_path).exists())
        is_video = (dance.get('type') == 'video' and video_path
                    and Path(video_path).exists())

        if is_video:
            self.pl_is_video = True
            self.video_player.setSource(QUrl.fromLocalFile(video_path))
            self.video_player.setPosition(0)
            if has_audio:
                self.video_audio_output.setMuted(True)
                self.audio_player.setSource(QUrl.fromLocalFile(audio_path))
                self.audio_player.setPosition(0)
                self.audio_player.play()
            else:
                self.video_audio_output.setMuted(False)
            self.video_player.play()
        else:
            self.pl_is_video = False
            if has_audio:
                self.audio_player.setSource(QUrl.fromLocalFile(audio_path))
                self.audio_player.setPosition(0)
                self.audio_player.play()
            else:
                self.pl_clock.restart()

        self.pose_preview_label.setVisible(True)
        self.btn_play_sel.setEnabled(False)
        self.btn_play_stop.setEnabled(True)

        names = dance_character_names(dance)
        n_chars = len(names)
        who = ""
        if n_chars > 1:
            if self.ref_char_idx == REF_AUTO_LEAD:
                who = tr('play_who_auto_lead')
            elif 0 <= self.ref_char_idx < len(names):
                who = tr('play_who_char',
                         n=self.ref_char_idx + 1,
                         name=names[self.ref_char_idx])
        self.lbl_play_status.setText(
            tr('play_playing', name=dance['name'], who=who)
        )

    def play_clock(self):
        if self.pl_is_video:
            t = self.video_player.position() / 1000.0
            if (self.pl_dance
                    and self.pl_dance.get('audio_path')
                    and Path(self.pl_dance['audio_path']).exists()
                    and self.audio_player.playbackState()
                        == QMediaPlayer.PlaybackState.PlayingState):
                audio_t = self.audio_player.position() / 1000.0
                # Агрессивная синхронизация: не даём разъехаться > 0.15 сек
                if abs(audio_t - t) > 0.15:
                    self.audio_player.setPosition(int(t * 1000))
            return t
        if (self.pl_dance and self.pl_dance.get('audio_path') and
                self.audio_player.playbackState() ==
                QMediaPlayer.PlaybackState.PlayingState):
            return self.audio_player.position() / 1000.0
        return self.pl_clock.elapsed() / 1000.0

    def play_score_pct(self):
        if self.pl_score_max <= 0:
            return 0.0
        return min(100.0, (self.pl_score_sum / self.pl_score_max) * 100.0)

    def play_score_display(self):
        return int(round(self.play_score_pct() * SCORE_MAX / 100.0))

    def score_play_frame(self):
        if not self.pl_dance or self.pl_finishing:
            return
        t = self.play_clock()
        try:
            duration = float(self.pl_dance.get('duration', 0) or 0)
        except Exception:
            duration = 0.0

        # Безопасный расчёт прогресса для QProgressBar (int32)
        self.progress.setValue(safe_pct_0_100(
            (t / duration * 100.0) if duration > 0 else 0.0
        ))

        frame = find_frame_at(self.pl_dance['frames'], t)
        if frame:
            poses = get_poses_from_frame(frame)
            if self.ref_char_idx == REF_AUTO_LEAD:
                idx = self.ref_lead_tracker.update(poses)
                if idx is not None and 0 <= idx < len(poses):
                    self.pl_target = poses[idx]
            else:
                self.pl_target = select_pose(poses, self.ref_char_idx)

        if (self.current_keypoints and frame and self.pl_target
                and t <= duration + 1):
            target_kp = extract_keypoints(self.pl_target)
            s = similarity_score(self.current_keypoints, target_kp)
            self.pl_last_score = s

            now = time.time()
            real_dt = now - self.pl_last_tick
            # Пропущено > 0.5 сек — не наказываем игрока, просто
            # перезапускаем таймер.
            if real_dt > 0.5:
                self.pl_last_tick = now
                real_dt = 0.0
            dt = max(0.005, min(0.2, real_dt)) if real_dt > 0 else 0.005
            self.pl_last_tick = now

            self.pl_score_sum += s * dt
            self.pl_score_max += 100 * dt

            if s >= THRESH_PERFECT:
                cat = 'perfect'
            elif s >= THRESH_GOOD:
                cat = 'good'
            elif s >= THRESH_OK:
                cat = 'ok'
            else:
                cat = 'miss'
            if cat != self.verdict_current:
                self.verdict_current = cat
                self._show_verdict(cat)

            if now - self.pl_stats_last_update >= 0.1:
                self.pl_stats_last_update = now
                st = self.pl_stats
                st['total'] += 1
                st['sum'] += s
                if s >= THRESH_PERFECT:
                    st['perfect'] += 1
                elif s >= THRESH_GOOD:
                    st['good'] += 1
                elif s >= THRESH_OK:
                    st['ok'] += 1
                else:
                    st['miss'] += 1

                if s >= THRESH_GOOD:
                    st['cur_combo'] += 1
                    if st['cur_combo'] > st['max_combo']:
                        st['max_combo'] = st['cur_combo']
                else:
                    st['cur_combo'] = 0

        if (t > duration + 0.5) or self._is_playback_finished():
            self._finish_play()

    def _finish_play(self):
        if self.pl_finishing:
            return
        self.pl_finishing = True
        try:
            st = self.pl_stats
            total = max(1, st['total'])
            score = self.play_score_display()
            avg = st['sum'] / total if total > 0 else 0.0

            char_label = None
            if self.pl_dance:
                names = dance_character_names(self.pl_dance)
                n = len(names)
                if n > 1:
                    if self.ref_char_idx == REF_AUTO_LEAD:
                        char_label = tr('play_auto_lead')
                    elif 0 <= self.ref_char_idx < n:
                        char_label = f"#{self.ref_char_idx + 1} · {names[self.ref_char_idx]}"

            stats_view = {
                'score': score, 'total': st['total'],
                'perfect': st['perfect'], 'good': st['good'],
                'ok': st['ok'], 'miss': st['miss'],
                'avg': avg, 'max_combo': st['max_combo'],
                'character_label': char_label,
            }

            if self.is_fullscreen:
                self.exit_fullscreen()

            self.stop_play()
            QTimer.singleShot(80, lambda: self._show_stats_dialog(stats_view))
        finally:
            self.pl_finishing = False

    def _show_stats_dialog(self, stats_view):
        dlg = StatsDialog(stats_view, parent=self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.raise_()
        dlg.activateWindow()
        dlg.exec()

    def stop_play(self):
        if not self.pl_active:
            return
        self.pl_active = False
        self._reset_verdict()
        self.audio_player.stop()
        self.video_player.stop()
        self.video_player.setSource(QUrl())
        self.video_audio_output.setMuted(False)
        self.pl_target = None
        self.current_video_image = None
        self.progress.setValue(0)
        self.pose_preview_label.setVisible(False)
        self.pose_preview_label.clear()
        self.pl_is_video = False
        self.btn_play_sel.setEnabled(True)
        self.btn_play_stop.setEnabled(False)
        self.lbl_play_status.setText("")

    # --- close ---
    def closeEvent(self, event):
        self.render_timer.stop()
        self._stop_camera()
        self.audio_player.stop()
        self.video_player.stop()
        if (self.video_processor is not None
                and self.video_processor.isRunning()):
            self.video_processor.wait(1500)
        event.accept()


# ============================================================
# Entry point
# ============================================================
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow{background:#0a0a1a;}
        QWidget{font-family:'Segoe UI',sans-serif;}
        QMessageBox{background:#16162e;color:#e8e8ff;}
        QMessageBox QLabel{color:#e8e8ff;font-size:13px;}
        QMessageBox QPushButton{background:#2a2a52;color:#fff;
            padding:6px 14px;border-radius:6px;border:none;min-width:70px;}
        QMessageBox QPushButton:hover{background:#3a3a72;}
        QCheckBox{color:#e8e8ff;spacing:8px;}
        QCheckBox::indicator{width:16px;height:16px;}
        QFileDialog{background:#16162e;color:#e8e8ff;}
        QInputDialog{background:#16162e;color:#e8e8ff;}
        QInputDialog QLineEdit{background:#0f0f24;color:#e8e8ff;
            border:1px solid #2a2a52;border-radius:6px;padding:8px;}
        QInputDialog QPushButton{background:#2a2a52;color:#fff;
            padding:6px 14px;border-radius:6px;border:none;}
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()