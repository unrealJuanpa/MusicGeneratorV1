import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import os
import sounddevice as sd
import soundfile as sf
from pytube import YouTube

import psutil
import subprocess

"""
NUEVO ENFOQUE:
- ENCODER DE 0.2 SEGUNDOS
- ANALIZA 102.42 SEGUNDOS CADA 0.017 SEGUNDOS
- 16 KHZ DE FRECUENCIA DE MUESTREO
- 3200 MUESTRAS CADA 0.017 SEGUNDOS
- LOS 512 LATENTES POR SEGUNDO SE REPARTEN EN CADA PARTICION = 100 VALORES (FINAL)
"""


sd.default.channels = 1

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class MARK(tf.keras.Model):
    def __init__(self, ckpt_name='', noise_stddev=1., compression_ndims=4096, shortm_states=2, longm_states=30):
        super().__init__()

        self.noise_stddev = noise_stddev
        self.compression_ndims = compression_ndims
        self.shortm_states = shortm_states
        self.longm_states = longm_states

        self.encoder_partitions = 8
        self.audio_samplerate = 16000
        self.latent_ndims = 512 // self.encoder_partitions

        sd.default.samplerate = self.audio_samplerate

        assert self.audio_samplerate % self.encoder_partitions == 0, 'Debe elegir un valor de encoder_partitions que sea divisor de audio samplerate'
        self.audio_ps = self.audio_samplerate // self.encoder_partitions

        self.audio_encoder = [ # 3200
            layers.Dense(2048, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(self.latent_ndims, activation='tanh')
        ]

        self.compression_layer = layers.Dense(self.compression_ndims, 'tanh')

        self.core_layers = [
            layers.Dense(4096, activation='tanh'),
            layers.Dense(4096, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(1024, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(2048, activation='tanh'),
            layers.Dense(4096, activation='tanh'),
            layers.Dense(8192, activation='tanh'),
            layers.Dense(self.audio_ps, activation='tanh')
        ]

        self(np.zeros([1, self.longm_states + self.shortm_states, self.audio_ps], dtype=np.float32))
        self.summary()

        self.loss = tf.keras.losses.MeanAbsoluteError()

        if ckpt_name != '':
            self.load_weights(ckpt_name)
            print('\n\nParametros cargados con exito!\n')

    def call_audio_encoder(self, x):
        assert len(x.shape) == 2 and x.shape[1] == self.audio_ps
        for idx in range(len(self.audio_encoder)):
            x = tf.concat([x, tf.random.normal([x.shape[0], 1], stddev=self.noise_stddev)], axis=1)
            x = self.audio_encoder[idx](x)
        return x

    def call_core(self, x):
        for idx in range(len(self.core_layers)):
            x = tf.concat([x, tf.random.normal([x.shape[0], 1], stddev=self.noise_stddev)], axis=1)
            x = self.core_layers[idx](x)
        return x

    def call(self, x):
        # x shape: [None, self.longm_states + self.shortm_states, self.audio_ps]
        assert x.shape[1:] == [self.longm_states + self.shortm_states, self.audio_ps]

        batch_size = x.shape[0]
        SHORTM = x[:, self.longm_states:, :]

        LONGM = self.call_audio_encoder(tf.reshape(x[:, :self.longm_states, :], (batch_size*self.longm_states, self.audio_ps)))
        LONGM = tf.reshape(LONGM, (batch_size, self.longm_states*self.latent_ndims))
        LONGM = self.compression_layer(LONGM)

        SHORTM = self.call_audio_encoder(tf.reshape(x[:, self.longm_states:, :], (batch_size*self.shortm_states, self.audio_ps)))
        SHORTM = tf.reshape(SHORTM, (batch_size, self.shortm_states*self.latent_ndims))

        LONGM = tf.concat([LONGM, SHORTM], axis=-1)

        for idx in range(len(self.core_layers)):
            LONGM = self.core_layers[idx](LONGM)
        return LONGM

    @tf.function
    def fitstep(self, X, Y):
        with tf.GradientTape() as tape:
            out = self(X)
            loss = self.loss(Y, out)

        g = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(g, self.trainable_variables))
        return loss

    @tf.function(jit_compile=True)
    def fitstep_compilled(self, X, Y):
        with tf.GradientTape() as tape:
            out = self(X)
            loss = self.loss(Y, out)

        g = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(g, self.trainable_variables))
        return loss

    def norm_path(self, path):
        tokens = [' ', '&', '(', ')', '{', '}', '[', ']']

        for token in tokens:
            path = path.replace(token, f'\{token}')
        return path

    def download_audio_from_youtube(self, links):
        if not os.path.exists(os.path.join(os.getcwd(), 'FitFiles')):
            os.makedirs(os.path.join(os.getcwd(), 'FitFiles'))

        if not os.path.exists(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles')):
            os.makedirs(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles'))

        if not os.path.exists(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles')):
            os.makedirs(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles'))

        print('\n')
        for i in links:
            print(f'Descargando video desde {i}')
            yt = YouTube(i)
            yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')
            yt[len(yt)//2].download(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles'))

        fnames = tuple(filter(lambda x: x.endswith('.mp4'), next(os.walk(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles')))[2]))

        print('\n\nExtrayendo audio de los videos...\n\n')

        for fname in fnames:
            audioname = fname.split('.')[0] + '.wav'
            audioname = self.norm_path(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles', audioname))
            videoname = self.norm_path(os.path.join(os.getcwd(), 'FitFiles', 'VideoFiles', fname))
            subprocess.call(f'ffmpeg -i {videoname} -ab 160k -ac 1 -ar 16000 -vn {audioname}', shell=True)

        print('\n\nDatos obtenidos con exito!\n\n')

    def fitOnFiles(self, links, optimizer, lr, epochs, batch_size, ckpt_iters, ckpt_name):
        self.opt = getattr(tf.keras.optimizers, optimizer)(lr)
        self.download_audio_from_youtube(links)

        X = np.zeros((batch_size, self.longm_states + self.shortm_states, self.audio_ps), dtype=np.float32)
        Y = np.zeros((batch_size, self.audio_ps), dtype=np.float32)
        fnames = tuple(filter(lambda x: x.endswith('.wav'), next(os.walk(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles')))[2]))

        idx_count = 0

        print(f'\n\nCiclo de entrenamiento iniciado... RAM usage: {psutil.virtual_memory().percent}%')

        for ep in range(1, epochs+1, 1):
            for fname in fnames:
                track = sf.SoundFile(os.path.join(os.getcwd(), 'FitFiles', 'AudioFiles', fname))
                assert track.samplerate == self.audio_samplerate

                duration = int(np.ceil((track.frames / track.samplerate) * self.encoder_partitions))

                for i in range(duration):
                    chunk = track.read(self.audio_ps)
                    chunk = np.concatenate((chunk, np.zeros((self.audio_ps - chunk.shape[0],), dtype=np.float32)), axis=0)

                    X[idx_count, :-1] = X[idx_count-1, 1:]
                    X[idx_count, -1] = Y[idx_count - 1]
                    Y[idx_count] = chunk

                    idx_count += 1

                    if idx_count >= batch_size:
                        print(f'Epoch {ep}/{epochs} | Loss {self.fitstep(X, Y)} | RAM usage: {psutil.virtual_memory().percent}%')
                        idx_count = 0

            print()

            if ep % ckpt_iters == 0:
                print('Guardando checkpoint...')
                self.save_weights(ckpt_name)
                print('Checkpoint guardado con exito!')

        print('Guardando parametros!')
        self.save_weights(ckpt_name)
        print('Parametros guardados con exito!\n')

    def load_wav(self, filename):
        wav = tf.io.read_file(filename)
        wav, sps = tf.audio.decode_wav(wav, desired_channels=1)
        assert sps == self.audio_samplerate, f'El audio debe ser de 16khz, no de {sps}'
        return wav[:, 0].numpy()

    def vtop(self, x):
        return x*(x>0)

    def save_wav(self, outarr, filename):
        assert len(outarr.shape) == 1
        outarr = tf.audio.encode_wav(outarr[..., np.newaxis], self.audio_samplerate)
        tf.io.write_file(filename, outarr)

    def generate(self, x, seconds, filename):
        seconds = seconds*self.encoder_partitions
        assert len(x.shape) == 1
        #x = np.concatenate((x, np.zeros((x.shape[0] % self.audio_samplerate), dtype=np.float32)), axis=0)
        x = np.reshape(x, (x.shape[0]//self.audio_ps, self.audio_ps))[np.newaxis, ...]

        x = x[:, -(self.longm_states + self.shortm_states):]
        x = np.concatenate((np.zeros((1, self.longm_states+self.shortm_states-x.shape[1] , self.audio_ps),dtype=np.float32), x), axis=1)

        outarr = np.zeros((seconds, self.audio_ps), dtype=np.float32)

        print(f'\n\nGenerando {seconds} de audio...\n')

        for i in range(seconds):
            outsec = self(x).numpy()[0]
            x[0, :-1] = x[0, 1:]
            x[0, -1] = np.copy(outsec)
            outarr[i] = np.copy(outsec)

            print(f'Generando {i+1} de {seconds} particiones...')

        print(f'\nExportando audio...')
        self.save_wav(outarr.flatten(), filename)
        print(f'\nAudio exportado con exito!')
