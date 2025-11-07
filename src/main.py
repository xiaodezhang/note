import json
from datetime import datetime
import soundfile as sf
import sounddevice as sd
import numpy as np
from loguru import logger
from PySide6.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QProgressBar, QPushButton, QSizePolicy, QToolButton, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QSize, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor, QFontMetrics, QIcon, QPainter
import sys
import whisperx
import torch
import hashlib

def get_file_sha256(file_path):
    sha256 = hashlib.sha256()
    # Open file in binary mode
    with open(file_path, "rb") as f:
        # Read and update hash in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def audio_to_lyric(path):
    # 1. 自动检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 2. 加载 Whisper 模型（选择 base / small / medium / large-v2）
    compute_type = "float16" if device.startswith("cuda") else "float32"
    model = whisperx.load_model("small", device=device, compute_type=compute_type)

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
    def __init__(self, lyrics, parent=None):
        super().__init__(parent)
        self._lyrics = lyrics
        self._current_index = 0

        self.line_height = 30
        self.visible_lines = 7

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(self.line_height * self.visible_lines)
        self.setMinimumWidth(400)

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
        self._current_index = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)

        # logger.warning(f'PAINT BEGIN, current_index: {self._current_index}----------------------------------------------')
        self._calculate_overlays()

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


class PlayPauseButton(QToolButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setIconSize(QSize(32, 32))
        self.update_icon()
        self.clicked.connect(self.update_icon)

    def update_icon(self):
        icon = QIcon("image/light/pause_circle.svg") if self.isChecked() else QIcon("image/light/play_circle.svg")
        self.setIcon(icon)

class PlayerWidget(QWidget):
    playing_progress = Signal(float)
    next_clicked = Signal()
    def __init__(self):
        super().__init__()
        self._stream = None

        layout = QHBoxLayout(self)
        self._play_pause_button = PlayPauseButton(self)
        self._next_button = QPushButton('下一句')
        self._current_button = QPushButton('播放当前语句')
        self._current_slow_button = QPushButton('播放当前语句(0.5倍速)')

        layout.addWidget(self._play_pause_button)
        layout.addWidget(self._current_button)
        layout.addWidget(self._current_slow_button)
        layout.addWidget(self._next_button)

        self._play_pause_button.toggled.connect(lambda: self.toggle_play())
        self._next_button.clicked.connect(self._on_next)
        # self._current_button.clicked.connect(self._on_current)

    def _on_next(self):
        self.next_clicked.emit()

    def play_at(self, sec):
        self._position = sec * self._samplerate
        if not self._playing:
            logger.debug('toggle_play')
            self.toggle_play()

    @property
    def audio_file(self):
        """The audio_file property."""
        return self._audio_file

    @audio_file.setter
    def audio_file(self, value):
        self._audio_file = value

        self._data, self._samplerate = sf.read(value)

        # 音频流
        self._stream = None
        self._playing = False
        self._position = 0  # 当前播放样本索引

    # 音频回调
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)

        if not self._playing:
            outdata.fill(0)
            return

        start = int(self._position)
        # target_data = self.data_slow if self.slow else self.data
        target_data = self._data
        end = start + frames
        if end > len(target_data):
            outdata[:len(target_data)-start] = target_data[start:]
            outdata[len(target_data)-start:] = 0
            self._position = len(target_data)
            raise sd.CallbackStop()
        else:
            outdata[:] = target_data[start:end]
            self._position += frames

        elapsed = start / self._samplerate
        self.playing_progress.emit(elapsed)

    # 播放/暂停
    def toggle_play(self):
        if self._playing:
            self._playing = False
            assert self._stream
            self._paused_start = self._stream.time

        else:
            if self._stream is None:
                channels = self._data.shape[1] if self._data.ndim > 1 else 1
                self._stream = sd.OutputStream(
                    samplerate=self._samplerate,
                    channels=channels,
                    callback=self.audio_callback,
                    blocksize=2 ** 5
                )
                self._stream.start()

            self._playing = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()

class ExtractLyricThread(QThread):
    finished2 = Signal(str, list)
    def run(self):
        self._lyric = audio_to_lyric(self._file_path)
        # logger.debug(f'lyrics: {self._lyric}')
        self.finished2.emit(self._file_path, self._lyric)

    def extract(self, path):
        self._file_path = path
        self.start()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

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
        self._next_flag = False
        self._lyric_widget= LyricWidget(self._lyrics)

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

        self.load()

    def _on_next(self):
        self._next_flag = True
        self._lyric_widget.current_index += 1
        segment = self._lyrics[self._lyric_widget.current_index + 1]
        self._player_widget.play_at(segment['start'])
        logger.debug(f'start: {segment["start"]}, current: {self._lyric_widget.current_index}')

    def _on_playing(self, elapsed):
        if self._lyrics:
            seg = self._lyrics[self._lyric_widget.current_index]
            next_seg = self._lyrics[self._lyric_widget.current_index + 1]
            self._last_end = seg['end'] + (next_seg['start'] - seg['end']) * 0.5
            self._next_start = next_seg['start']

            logger.debug(f'seg end: {seg["end"]}, next start: {next_seg["start"]}')


            logger.debug(f'elapsed: {elapsed}, last_end: {self._last_end}, current_index: {self._lyric_widget.current_index}')

            # logger.debug(f'current_index: {self._lyric_widget.current_index}')

            if elapsed > self._last_end:
                if self._next_flag:
                    self._player_widget.toggle_play()
                    logger.debug(f'toggle_play')
                    self._next_flag = False

            # if elapsed > self._next_start:
            #     self._lyric_widget.current_index += 1

        # for i, seg in enumerate(self._lyrics):
        #     if elapsed >= seg['start'] and elapsed <= seg['end']:
        #         self._lyric_widget.current_index = i
        #         self._last_end = seg['end']
        #         break

    def _on_extract_finished(self, file, lyrics):
        self._progress.hide()
        # logger.debug(f'lyrics: {lyrics}')
        self._lyrics = lyrics
        self._lyric_widget.lyrics = lyrics
        self._audios.append((file, lyrics))

    def _on_open_audio(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", ".", "Audio Files (*.mp3 *.wav *.flac)"
        )
        if not file:
            return

        self._player_widget.audio_file = file
        self._extract_lyric_thread.extract(file)
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
                    self._player_widget.audio_file, self._lyrics = self._audios[0]
                    self._lyric_widget.lyrics = self._lyrics

        except: ...

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(500, 200)
    window.show()

    app.exec()

    window.save()
