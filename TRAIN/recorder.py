import sounddevice as sd
from scipy.io.wavfile import write
from os import chdir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n_times", help="how many datapoints you want to record")

args = parser.parse_args()

def record_wake_word(savepath, start_num:int , n_times: int, seconds, prompt: bool = False):
    chdir(savepath)
    input("something:")
    for i in range(start_num, start_num+n_times):
        fs = 16_000
        print("recording...")
        recording = sd.rec(int(fs*seconds), samplerate=fs, channels=2)
        sd.wait()
        write(str(i) + ".wav", fs, recording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ") if prompt else print(f"recorded {i}/{start_num+n_times}")


record_wake_word("data-2seconds/wakeword/", 0, args.n_times, 2, prompt=True)