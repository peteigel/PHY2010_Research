from numpy import *
from matplotlib import pyplot
from scipy import signal

def plotSamples (a, b, Fs, trim=False):
    start = int(Fs/8) if trim else 0
    end = int(Fs/4) if trim else len(a)
    time = linspace(start/Fs, end/Fs, end - start)

    pyplot.plot(
        time,
        a[start:end],
        label="Target Signal"
    )

    pyplot.plot(
        time,
        b[start:end],
        label="Output Signal"
    )
    pyplot.title("Waveform Comparison")
    pyplot.xlabel("Time (seconds)")
    pyplot.ylabel("Amplitude")
    pyplot.show()

def plotSpectrum (a, b, Fs):
    fa, pa = signal.welch(a, Fs, scaling='spectrum')
    fb, pb = signal.welch(b, Fs, scaling='spectrum')

    pyplot.loglog(
        fa,
        pa,
        label="Target Signal"
    )

    pyplot.plot(
        fb,
        pb,
        label="Output Signal"
    )

    pyplot.title("Spectrum Comparison")
    pyplot.xlabel("Frequency (Hz)")
    pyplot.ylabel("Power")
    pyplot.show()
