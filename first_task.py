from enum import Enum
import math

class Signal_type(Enum):
    Time = 0
    Freq = 1

class Signal() :
    analog_freq = 0
    sampling_freq = 0
    amp = 0
    phase_shift = 0
    sin : bool = True
    periodic : bool = 0
    signal_type : Signal_type = Signal_type.Time
    s = []
    
    def __init__(self,sin: bool ,periodic : bool,signal_type : Signal_type, amp,sampling_freq, analog_freq,phase_shift):
        self.analog_freq = analog_freq
        self.sampling_freq = sampling_freq
        self.sin = sin
        self.amp = amp
        self.phase_shift =phase_shift
        self.periodic = periodic 
        self.signal_type = signal_type 
        w = 2 * math.pi * (self.analog_freq / self.sampling_freq)
        if sin :
            self.s = [self.amp * math.sin( w * i + self.phase_shift) for i in range(0,self.sampling_freq) ]
        else:
            self.s = [self.amp * math.cos( w * i + self.phase_shift) for i in range(0,self.sampling_freq) ]

    def get_signal(self):
        return self.s

    # def export(self,file_name):
    #     with open(file_name, 'w') as f:
    #         f.writek


sin_sig = Signal(True,False,Signal_type.Time,3,720,360,1.96349540849362)
cos_sig = Signal(False,False,Signal_type.Time,3,500,200,2.35619449019235)

s = sin_sig.get_signal()
c = cos_sig.get_signal()

for i in range (len(c)) : 
    print(f'{i}  {c[i]}')
