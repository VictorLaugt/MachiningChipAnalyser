from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Sequence
    Signal = Sequence[float]


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class GraphAnimation:
    def __init__(self, signal_seqs: Sequence[Sequence[Signal]], xlabel='', ylabel=''):
        self.n_frames = 0
        self.n_signal = 0
        n_values = 0
        for sig_seq in signal_seqs:
            self.n_frames = max(self.n_frames, len(sig_seq))
            self.n_signal += 1
            for sig in sig_seq:
                n_values = max(n_values, len(sig))

        self.data = np.zeros((self.n_frames, n_values, self.n_signal), dtype=np.float64)
        for s, sig_seq in enumerate(signal_seqs):
            for f, sig in enumerate(sig_seq):
                self.data[f, :len(sig), s] = sig

        self.fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        self.frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        self.lines = ax.plot(self.data[0, :, :], '-x')
        self.artists = [*self.lines, self.frame_text]

        self.state_play = True
        self.state_skip_frame = False
        self.frame_index = 0

    def _update(self, _):
        if self.state_play or self.state_skip_frame:
            self.state_skip_frame = False
            self.frame_index = (self.frame_index + 1) % self.n_frames
            self.frame_text.set_text(f'frame: {self.frame_index+1}')
            for i in range(self.n_signal):
                self.lines[i].set_ydata(self.data[self.frame_index, :, i])
        return self.artists

    def _on_press(self, event):
        if event.key == ' ':
            self.state_play = not self.state_play
        elif event.key == '6':
            self.state_skip_frame = True
        elif event.key == '4':
            self.frame_index = (self.frame_index - 2) % self.n_frames
            self.state_skip_frame = True

    def play(self):
        _animation = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=30,
            blit=True,
            frames=self.n_frames
        )
        self.fig.canvas.mpl_connect('key_press_event', self._on_press)
        plt.show()
