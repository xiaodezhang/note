import json
import librosa
from pathlib import Path
import soundfile as sf
import sounddevice as sd
from loguru import logger
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QProgressBar, QPushButton, QSizePolicy, QToolButton, QWidget, QVBoxLayout
from PySide6.QtCore import QObject, QSize, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QFontMetrics, QIcon, QPainter, QWheelEvent
import sys
import whisperx
import torch
import hashlib
from qt_material import apply_stylesheet

def url(path):
    theme = 'light'
    return str(Path('image') / theme / path)

def get_slow_file(file: str) -> Path:
    f = Path(file)
    return f.parent / (f.stem + 'slow.wav')

def get_file_sha256(file_path):
    sha256 = hashlib.sha256()
    # Open file in binary mode
    with open(file_path, "rb") as f:
        # Read and update hash in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def to_slow_audio(path: Path):
    y, sp = librosa.load(path, sr=None)
    y_slow = librosa.effects.time_stretch(y, rate=0.5)
    sf.write(path.parent / (path.stem + 'slow.wav'), y_slow, sp)

def audio_to_lyric(path):
    # 1. 自动检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 2. 加载 Whisper 模型（选择 base / small / medium / large-v2）
    compute_type = "float16" if device.startswith("cuda") else "float32"
    model = whisperx.load_model("small.en", device=device, compute_type=compute_type)

    # 3. 执行语音识别
    result = model.transcribe(path)

    print("Detected language:", result["language"])
    print("Segments before alignment:", len(result["segments"]))

    # 4. 对齐（提高时间精度）
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned_result = whisperx.align(
        result["segments"], model_a, metadata, path, device
    )

    # logger.debug(f'result: {aligned_result}')

    return aligned_result['segments']

class LyricWidget(QWidget):
    play_start_changed = Signal(float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lyrics = []
        self._current_index = 0

        self.line_height = 30
        self.visible_lines = 7

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(self.line_height * self.visible_lines)
        self.setMinimumWidth(400)

    def wheelEvent(self, event: QWheelEvent, /) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.current_index -= 1

        else:
            self.current_index += 1

        self.play_start_changed.emit(self._lyrics[self._current_index]['start'])
        super().wheelEvent(event)

    @property
    def lyrics(self):
        """The lyrics property."""
        return self._lyrics

    @lyrics.setter
    def lyrics(self, value):
        self._lyrics = value
        self._current_index = 0

        self.update()

    def _calculate_overlays(self):
        self._overlays = []

        overlay = 0

        font = QFont("Arial", 12)
        fm = QFontMetrics(font)
        for i, line in enumerate(self._lyrics):
            row = ''
            row_overlay = 0
            for w in line['words']:
                ch = w['word']
                if fm.horizontalAdvance(row+ ch) > self.rect().width() -10:
                    row = ch + ' '
                    row_overlay += 1

                else:
                    row += ch + ' '

            self._overlays.append(overlay)
            # logger.debug(f'row: {row}, overlay: {overlay}')
            overlay += row_overlay + 1

        # logger.debug(f'overlays: {self._overlays}')

    def next_line(self):
        self._current_index = min(len(self._lyrics)-1, self._current_index + 1)
        self.update()

    @property
    def current_index(self):
        """The current_index property."""
        return self._current_index

    @current_index.setter
    def current_index(self, value):
        if self._current_index == value:
            return

        self._current_index = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)

        # logger.warning(f'PAINT BEGIN, current_index: {self._current_index}----------------------------------------------')
        self._calculate_overlays()

        if self._overlays:
            current_overlay = self._overlays[self._current_index]
            for i, (line, overlay) in enumerate(zip(self._lyrics, self._overlays)):
                y = (overlay - current_overlay) * self.line_height + self.height() // 2
                if 0 <= y <= self.rect().height():
                    visiable = True

                else:
                    visiable = False

                if i == self._current_index:
                    painter.setPen(QColor("red"))
                    # font = QFont("Arial", 14, QFont.Weight.Bold)
                    font = QFont("Arial", 12)
                else:
                    painter.setPen(QColor("black"))
                    font = QFont("Arial", 12)

                painter.setFont(font)

                fm = QFontMetrics(font)
                row = ''
                row_overlay = 0
                for w in line['words']:
                    ch = w['word']
                    if fm.horizontalAdvance(row+ ch) > self.rect().width() -10:
                        # y = (i - self._current_index + overlay + row_overlay) * self.line_height + self.height() // 2
                        y = (overlay - current_overlay + row_overlay) * self.line_height + self.height() // 2
                        if visiable:
                            painter.drawText(10, int(y + self.line_height*0.8), row)
                            # logger.debug(f'overlay: {overlay}, row_overlay: {row_overlay}, current_overlay: {current_overlay}, i: {i}, y: {overlay - current_overlay + row_overlay}, text: {row}')
                        row = ch + ' '
                        row_overlay += 1

                    else:
                        row += ch + ' '

                if row:
                    if visiable:
                        # y = (i - self._current_index + overlay + row_overlay) * self.line_height + self.height() // 2
                        y = (overlay - current_overlay + row_overlay) * self.line_height + self.height() // 2
                        painter.drawText(10, int(y + self.line_height*0.8), row)
                    # logger.debug(f'overlay: {overlay}, row_overlay: {row_overlay}, current_overlay: {current_overlay}, i: {i}, y: {overlay - current_overlay + row_overlay}, text: {row}')


class Button(QToolButton):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setIconSize(QSize(32, 32))
        self.setIcon(QIcon(url(path)))

class CheckedButton(QToolButton):
    def __init__(self, path1, path2, parent=None):
        super().__init__(parent)
        self._path1 = path1
        self._path2 = path2
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setIconSize(QSize(32, 32))

        self.update_icon()
        self.toggled.connect(self.update_icon)

    def update_icon(self):
        icon = QIcon(url(self._path1)) if self.isChecked() else QIcon(url(self._path2))
        self.setIcon(icon)

class Stream(QObject):
    playing_progress = Signal(float)
    playing_changed = Signal(bool)
    def __init__(self):
        super().__init__()
        self._stream = None
        self._playing = False
        self._position = 0  # 当前播放样本索引

    def set_audio_file(self, value):
        self._data, self._samplerate = sf.read(value)

        # 音频流
        self._stream = None
        self._playing = False
        self._position = 0  # 当前播放样本索引

    def play_at(self, sec):
        if not self._playing:
            self.set_second(sec)
            self.toggle_play()

    def set_second(self, sec):
        self._position = sec * self._samplerate

    @property
    def playing(self):
        """The playing property."""
        return self._playing

    @playing.setter
    def playing(self, value):
        if self._playing == value:
            return

        self._playing = value
        self.playing_changed.emit(value)

    # 音频回调
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)

        if not self._playing:
            outdata.fill(0)
            return

        start = int(self._position)
        end = start + frames
        if end > len(self._data):
            outdata[:len(self._data)-start] = self._data[start:]
            outdata[len(self._data)-start:] = 0
            self._position = len(self._data)
            raise sd.CallbackStop()

        else:
            if self._data.ndim == 1 and outdata.ndim == 2:
                outdata[:, 0] = self._data[start:end]
            elif self._data.ndim == 2 and outdata.ndim == 2:
                outdata[:] = self._data[start:end]
            self._position += frames

        elapsed = start / self._samplerate
        self.playing_progress.emit(elapsed)

    # 播放/暂停
    def toggle_play(self):
        if self._playing:
            self.playing = False
            assert self._stream
            self._paused_start = self._stream.time

        else:
            if self._stream is None:
                channels = self._data.shape[1] if self._data.ndim > 1 else 1
                # logger.debug(f'channels: {channels}')
                self._stream = sd.OutputStream(
                    samplerate=self._samplerate,
                    channels=channels,
                    callback=self.audio_callback,
                    # blocksize=2 ** 5
                )
                self._stream.start()

            self.playing = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()

class PlayerWidget(QWidget):
    playing_progress = Signal(float, bool)
    next_clicked = Signal(bool)
    previous_clicked = Signal(bool)
    current_clicked = Signal(bool)
    def __init__(self):
        super().__init__()
        self._normal_stream = Stream()
        self._slow_stream = Stream()
        self._stream = self._normal_stream

        layout = QHBoxLayout(self)
        self._play_pause_button = CheckedButton('pause.svg', 'play_arrow.svg')
        self._current_button = Button('refresh.svg')
        self._next_button = Button('skip_next.svg')
        self._previous_button = Button('skip_previous.svg')
        self._mode_button = CheckedButton('speed_0_5.svg', '1x_mobiledata.svg')

        layout.addWidget(self._mode_button)
        layout.addWidget(self._play_pause_button)
        layout.addWidget(self._current_button)
        layout.addWidget(self._previous_button)
        layout.addWidget(self._next_button)

        self._play_pause_button.toggled.connect(self._on_play)
        self._next_button.clicked.connect(lambda: self.next_clicked.emit(self._mode_button.isChecked()))
        self._previous_button.clicked.connect(lambda: self.previous_clicked.emit(self._mode_button.isChecked()))
        self._current_button.clicked.connect(lambda: self.current_clicked.emit(self._mode_button.isChecked()))
        self._mode_button.toggled.connect(self._on_mode_toggle)

        self._normal_stream.playing_progress.connect(lambda elapsed: self.playing_progress.emit(elapsed, False))
        self._slow_stream.playing_progress.connect(lambda elapsed: self.playing_progress.emit(elapsed, True))

        self._normal_stream.playing_changed.connect(self._on_playing_change)
        self._slow_stream.playing_changed.connect(self._on_playing_change)

    def _on_mode_toggle(self, checked):
        self._stream = self._slow_stream if checked else self._normal_stream

    def set_play_start(self, sec):
        self._normal_stream.set_second(sec)
        self._slow_stream.set_second(sec)

    def _on_playing_change(self, flag):
        self._play_pause_button.setChecked(flag)

    def set_audio_file(self, value):
        self._normal_stream.set_audio_file(value)
        self._slow_stream.set_audio_file(str(get_slow_file(value)))

    def play_at(self, sec, slow = False):
        if slow:
            self._stream = self._slow_stream

        else:
            self._stream = self._normal_stream

        self._stream.play_at(sec)

    def _on_play(self, checked):
        if self._stream.playing != checked:
            self._stream.toggle_play()

    def stop(self):
        self._normal_stream.stop()
        self._slow_stream.stop()

    def toggle_play(self):
        self._stream.toggle_play()

class ExtractLyricThread(QThread):
    finished2 = Signal(str, list)
    def run(self):
        self._lyric = audio_to_lyric(self._file_path)
        to_slow_audio(Path(self._file_path))
        # logger.debug(f'lyrics: {self._lyric}')
        self.finished2.emit(self._file_path, self._lyric)

    def extract(self, path: str):
        self._file_path = path
        self.start()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Learning English')
        self.setWindowIcon(QIcon(url('book_ribbon.svg')))

        layout = QVBoxLayout(self)
        open_audio_button = QPushButton('打开音频')
        self._player_widget = PlayerWidget()
        self._progress = QProgressBar(self)
        self._extract_lyric_thread = ExtractLyricThread()

        self._progress.setMaximumHeight(1)
        self._progress.setMaximum(0)
        self._progress.hide()

        # self._lyrics = audio_to_lyric('media/audio.mp3')
        self._audios = []
        self._lyrics = []
        self._segment_flag = False
        self._lyric_widget= LyricWidget()

        layout.addWidget(self._progress)
        layout.addWidget(open_audio_button)
        layout.addWidget(self._lyric_widget)
        layout.addWidget(self._player_widget)

        # 模拟歌词播放
        def advance_line():
            self._lyric_widget.next_line()

        timer = QTimer(self)
        timer.timeout.connect(advance_line)
        # timer.start(1500)  # 每1.5秒高亮下一行

        open_audio_button.clicked.connect(self._on_open_audio)
        self._extract_lyric_thread.finished2.connect(self._on_extract_finished)
        self._player_widget.playing_progress.connect(self._on_playing)
        self._player_widget.next_clicked.connect(self._on_next)
        self._player_widget.previous_clicked.connect(self._on_previous)
        self._player_widget.current_clicked.connect(self._on_current)
        self._lyric_widget.play_start_changed.connect(self._on_play_start_change)

        self.load()

    def _on_play_start_change(self, value):
        self._player_widget.set_play_start(value)

    def _on_current(self, slow):
        self._segment_flag = True
        segment = self._lyrics[self._lyric_widget.current_index]
        start = segment['start'] * 2 if slow else segment['start']
        self._player_widget.play_at(start, slow)

    def _on_previous(self, slow):
        self._segment_flag = True
        self._lyric_widget.current_index -= 1
        segment = self._lyrics[self._lyric_widget.current_index]
        start = segment['start'] * 2 if slow else segment['start']
        self._player_widget.play_at(start, slow)

    def _on_next(self, slow):
        self._segment_flag = True
        self._lyric_widget.current_index += 1
        segment = self._lyrics[self._lyric_widget.current_index]
        start = segment['start'] * 2 if slow else segment['start']
        self._player_widget.play_at(start, slow)
        # logger.debug(f'start: {segment["start"]}, current: {self._lyric_widget.current_index}')

    def _on_playing(self, elapsed, slow):
        if self._lyrics:
            current = self._lyric_widget.current_index
            # logger.debug(f'current: {current}')
            if current < len(self._lyrics) - 1:
                seg = self._lyrics[current]
                next_seg = self._lyrics[self._lyric_widget.current_index + 1]

                if slow:
                    self._last_end = next_seg['start'] + seg['end']
                    self._next_start = next_seg['start'] * 2

                else:
                    self._last_end = (next_seg['start'] + seg['end']) * 0.5
                    self._next_start = next_seg['start']

                if elapsed > self._last_end:
                    if self._segment_flag:
                        self._player_widget.toggle_play()
                        # logger.debug(f'toggle_play')
                        self._segment_flag = False

                if elapsed > self._next_start:
                    self._lyric_widget.current_index += 1

            else:
                self._player_widget.toggle_play()


    def _on_extract_finished(self, file, lyrics):
        self._progress.hide()
        # logger.debug(f'lyrics: {lyrics}')
        self._lyrics = lyrics
        self._lyric_widget.lyrics = lyrics
        self._audios.append((file, lyrics))
        self._player_widget.set_audio_file(file)

    def _on_open_audio(self):
        self._file, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", ".", "Audio Files (*.mp3 *.wav *.flac)"
        )
        if not self._file:
            return

        self._extract_lyric_thread.extract(self._file)
        self._progress.show()

    def save(self):
        self._player_widget.stop()
        with open('audio.json', 'w', encoding='utf-8') as f:
            # json.dump(self._lyrics, f)
            json.dump(self._audios, f)

    def load(self):
        try:
            with open('audio.json', 'r', encoding='utf-8') as f:
                self._audios = json.load(f)
                if self._audios:
                    audio_file, self._lyrics = self._audios[0]
                    self._player_widget.set_audio_file(audio_file)
                    self._lyric_widget.lyrics = self._lyrics

        except: ...

if __name__ == "__main__":
    app = QApplication(sys.argv)


    theme = 'light'
    theme_name = theme  + '_theme.xml'
    apply_stylesheet(
        app
        , theme=theme_name
        , invert_secondary = False if theme == 'dark' else True
        , css_file='style/mystyle.css'
    )

    window = MainWindow()
    window.resize(500, 200)
    window.show()

    app.exec()

    window.save()
