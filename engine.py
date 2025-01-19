import pyaudio
import time
import threading
import torch
import wave
import torchaudio
from model import StackedConvNet


class FEATURE_INIT(torch.nn.Module):
  def __init__(self, sample_rate):
    super().__init__()
    self.transforms = torchaudio.transforms.MFCC(
      sample_rate=sample_rate,
      n_mfcc=80,
      melkwargs={'n_mels': 80, 'win_length': 160, 'hop_length': 80})
    
    
  def __call__(self, x):
    print(x.shape)
    x = self.transforms(x)

    return x


class Listener:

    def __init__(self, model_file, sample_rate=16_000, record_seconds=2):
        self.chunk = 1000
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)
        
        self.queue = []

        self.model = StackedConvNet(
                    in_channels=80,
                    intermediate_channels=128,
                    out_channels=8,
                    pool_size=45,
                    embed_dim=15,
                    num_layers=4
                )
        self.model.load_state_dict(torch.load(model_file))
        self.sampler = FEATURE_INIT(sample_rate=self.sample_rate)
        self.FLAG = False

        self.fname = "wakeword_temp"
        

    def listen(self):
        while self.FLAG != True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            self.queue.append(data)
            time.sleep(0.01)


    def save(self):
        wf = wave.open(self.fname, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(self.queue[:self.sample_rate*self.record_seconds]))
        wf.close()

        waveform, _ = torchaudio.load(self.fname)
        return  waveform.mean(0)


    def make_prediction(self):
        while True:
            with torch.no_grad():
                audio_data = self.sampler(self.save()).unsqueeze(0)
                pred = self.model(audio_data)

                if pred.item() > 0.9:
                    self.FLAG = True
                    print("wakeword found", pred.item())
                    break

                else:
                    del self.queue[0]
                    print("no wakeword found", pred.item(), len(self.queue))
                    time.sleep(0.1) if len(self.queue) < 40 else time.sleep(0.01)

    def run(self):
        ListenThread = threading.Thread(target=self.listen, daemon=True)
        ListenThread.start()

        time.sleep(2)
        PredThread = threading.Thread(target=self.make_prediction, daemon=True)
        PredThread.start()

        print("\nWake Word Engine is now listening... \n")
        
        PredThread.join()
        ListenThread.join()

Listener("model.pt").run()



