from MARK import MARK
import os
import numpy as np


ckpt_file = 'MARK_PARTITIONS.h5'

m = MARK(ckpt_file, shortm_states=2, longm_states=240)

m.fitOnFiles(links = ['https://www.youtube.com/watch?v=iagQFFUPxAM&ab_channel=ElvisPresley-Topic'],
             optimizer = 'Adam',
             lr = 0.00003,
             epochs = 1000,
             batch_size = 768,
             ckpt_iters = 50,
             ckpt_name = ckpt_file)
"""

x = m.load_wav(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles', 'Flaming Star.wav'))

desface = 0
end_frame = 128*16000
seconds_to_gen = 120

x = x[desface:end_frame+desface]

#input(x.shape)

m.noise_stddev = 4.

m.save_wav(x, 'input_flaming_star.wav')
m.generate(x, seconds=seconds_to_gen, filename='output_flaming_star.wav')
"""
