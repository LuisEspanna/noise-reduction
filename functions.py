import tensorflow as tf
import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from librosa.display import specshow
from tensorflow.python.keras import backend
import tensorflow_io as tfio
import IPython.display as ipd
import random
from tensorflow.python.keras.layers import Input, Conv2D, LeakyReLU, MaxPooling2D, Dropout, concatenate, UpSampling2D, Conv2DTranspose


BatchNormalization = tf.nn.batch_normalization

TOP_DB = 80.0
# Estandarización de duración
MAX_DUR = 6
# Normalización del espectrograma
NORMALIZE = True
# Remuestreo de frecuencia
RESAMPLING_FREQ = 22050
# Estandarización de duración
MAX_DUR = 6
SAMPLES = MAX_DUR * RESAMPLING_FREQ
N_FFT = 2**10 # Muestras en cada ventana 2**10=1024
INPUT_WIDTH = int(1+(N_FFT/2))
OVERLAPPING = 1/4 # Traslape entre ventanas
HOP_LENGTH = int(np.ceil(N_FFT*OVERLAPPING)) # Número de muestras de audio entre ventanas STFT adyacentes.
INPUT_HEIGHT = int( np.ceil( SAMPLES / HOP_LENGTH ) )
RESCALE_WIDTH = int(2**8) # 2**8=256. 2**9=512
RESCALE_HEIGHT = RESCALE_WIDTH
CHANNELS = 1
TF_ENABLE_ONEDNN_OPTS=0


random.seed(7)
np.random.seed(7)
tf.random.set_seed(7)


def pad_trunc_audio(audio, sampling_freq, max_dur):
    '''
    Esta función rellena o corta la duración del audio dependiendo del argumento de duración
    máxima: max_dur. El relleno o padding de audio se hace con un vector de ceros, lo cual
    equivale a silencio en audio y por lo tanto no afecta el espectograma del audio original.
    
    Input:
    - audio: Audio de entrada. Tensor de Tensorflow
    - sampling_freq: Frecuencia de muestreo del audio. int [Hz]
    - max_dur: Máxima duración permitida. int [s]
    Output:
    - audio_proc: Audio procesado. Tensor de Tensorflow,
    '''
    # sampling_freq = tf.get_static_value(sampling_freq)
    # max_dur = tf.get_static_value(max_dur)
    
    samples = len(audio) # Número de muestras del audio de entrada.
    max_samples = max_dur * sampling_freq # Mayor número de muestras permitida.
    
    samples = tf.cast(samples, tf.int64) # Se convierte a entero.
    max_samples = tf.cast(max_samples, tf.int64) # Se convierte a entero.
    sampling_freq = tf.cast(sampling_freq, tf.int64) # Se convierte a entero.
    
    # Si la duración del audio es mayor a la máxima duración seleccionada.
    if (samples/sampling_freq > max_dur):
        # El vector de audio original se recorta desde el inicio hasta la máxima
        # muestra permitida.
        audio_proc = audio[:max_samples]
        
    # Si la duración del audio es menor a la máxima duración seleccionada.
    elif(samples/sampling_freq < max_dur):
        # Se calcula el número de muestras de relleno que se agregan al audio original.
        padding_samples = max_samples - samples
        # Se crea la matriz de ceros (equivalente a silencio en audio) que se concatena
        # al vector del audio original.
        padding_matrix = tf.zeros([padding_samples], tf.float32)
        # Se concatena el audio original con la matriz de relleno.
        audio_proc = tf.concat((audio, padding_matrix), 0)
        
    # Si la duración del audio es igual a la máxima duración seleccionada.
    else:
        # Se retorna el mismo audio de entrada.
        audio_proc = audio

    return audio_proc


def tensor_to_nparray(tensor):
    '''
    Esta función convierte un objeto tipo tensor de Tensorflow en un objeto tipo array de NumPy.
    Es util para graficar el audio y calcular la STFT mediante la biblioteca Librosa.
    
    Input:
    - tensor: Objeto tipo tensor
    Output:
    - np_array: Objeto tipo array de numpy
    '''
    # Se convierte el objeto tipo Tensor a un array de numpy
    np_array = tensor.numpy()
    # Se disminuye la dimensión del vector
    np_array = np.squeeze(np_array)
    
    return np_array


def train_test_val_split(dataset, percent, train_percent, val_percent, test_percent):
    '''
    Función para dividir el conjunto de datos en entrenamiento, validación y evaluación.
    
    Parámetros:
    - dataset: Conjunto de entrada. Tensor de TensorFlow
    - training_size: Tamaño del conjunto de entrenamiento deseado.
    - validation_size: Tamaño del conjunto de validación deseado.
    (El tamaño del conjunto de evaluación se calcula automaticamente).
    Returns:
    - train_set: Conjunto de entrenamiento.
    - val_set: Conjunto de validación.
    - test_set: Conjunto de evaluación.
    '''
    size = dataset.cardinality().numpy() * percent

    train_size = tf.math.rint(tf.math.multiply(size, tf.cast(train_percent, tf.float32)))
    val_size = tf.math.rint(tf.math.multiply(size, tf.cast(val_percent, tf.float32)))
    test_size = tf.math.rint(tf.math.multiply(size, tf.cast(test_percent, tf.float32)))
    
    return tf.cast(train_size, tf.int64), tf.cast(val_size, tf.int64), tf.cast(test_size, tf.int64)



def get_mel_spectrogram(senal, fs=48000, n_fft=1024, hop_length=256, window='hann', n_mels=128, verbose=False):
    '''
    Esta función calcula el espectrograma de la señal de entrada en escala MEL de acuerdo a los argumentos dados. 
    
    Parámetros:
    - senal: Señal de entrada
    - fs: Frecuencia de Muestreo
    - n_ftt: Nro. de muestras para cada ventana donde se va a realizar la stft. Recomendación: Potencia de dos.
    - hop_length: Nro. muestras que se va a mover la ventana en cada desplazamiento (hop: salto)
    - window: Tipo de ventana.
    - n_mels: Número de filtros de mel
    - verbose: Activar o desactivar el debugging de la STFT.
    Returns:
    - log_mel_spec:  Magnitud del espectograma de la señal de entrada mapeado a escala MEL.
    '''
    # Se calcula la magnitud del espectograma de la señal de entrada, y lo mapea a escala MEL.
    mel_spec = lb.feature.melspectrogram(y=tensor_to_nparray(senal),
                                         sr=fs,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         window=window,
                                         n_mels=n_mels)
    # Se convierte la escala a decibeles
    log_mel_spec = lb.power_to_db(mel_spec, ref=np.max)
    
    # Se aumenta la dimensión para que el algoritmo lo tome como imagen de 1 canal
    log_mel_spec = tf.expand_dims(log_mel_spec, axis=-1);

    # Conversión de los arrays a tensores de TensorFlow
    log_mel_spec = tf.convert_to_tensor(log_mel_spec, dtype=tf.float32)

    # Datos de la transformada
    if verbose == True:
        print(f'Nro. de muestras dentro de cada ventana: {n_fft}')
        print(f'Solapamiento entre ventanas: {(hop_length/n_fft)*100} %')
        print(f'Duración de cada ventana: {(n_fft/fs) * 1000:.3f} ms')
        print(f'Dimensiones de la Mátriz resultante: {log_mel_spec.shape}')
        print(f'Resolución en frecuencia del espectograma: {log_mel_spec.shape[0]} divisiones.')
        print(f'Resolución en tiempo / Número de ventanas a lo largo de la serie de tiempo: {log_mel_spec.shape[1]} ventanas.')
            
    return log_mel_spec


def plot_time_audio(audio, fs, width=13, height=4, title='Grafica del audio en el dominio del Tiempo', save_plot=False):
    '''
    Esta función grafica la señal de audio en el tiempo, de acuerdo
    a la frecuencia de muestreo dada.
    
    Parámetros:
    - audio: Señal de audio a graficar.
    - fs: Frecuencia de muestreo del audio.
    - width: Ancho de la Figura.
    - height: Altura de la Figura.
    - title: Titulo de la Figura.
    - save_plot: Guardar Figura.
    Returns:
    - N/A
    '''
    audio_samples = len(audio)
    t = np.arange(0, audio_samples/fs, 1/fs)
    
    if len(t) > audio_samples:
        t = t[:-1]

    # Plotear Figura
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    fig.patch.set_facecolor('white')
    ax.plot(t, audio);
    ax.set_title(title)
    ax.set_xlabel('Tiempo [s]')
    ax.set_xlim([0, MAX_DUR])
    ax.set_ylabel('Amplitud')
    ax.grid(True)
    plt.tight_layout()
    
    # Guardar Figura
    if save_plot == True:
        fmt = 'png'
        file_name = f'{title}.{fmt}'
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0, dpi=300)


def plot_spectrogram(senal, sampling_freq, hop_length, width=5, height=5, title='Espectrograma', y_axis=None, cmap='inferno', save_plot=False):
    '''
    Esta función grafica el espectrograma de la señal de audio.
    
    Parámetros:
    - senal: Señal de audio a graficar.
    - sampling_freq: Frecuencia de muestreo del audio.
    - hop_length: Nro. muestras que se va a mover la ventana en cada desplazamiento (hop: salto)
    - width: Ancho de la grafica.
    - height: Altura de la grafica.
    - title: Titulo de la Figura.
    - cmap: Colores del mapa de calor de la Figura.
    - save_plot: Guardar Figura.
    Returns:
    - N/A
    '''
    if tf.is_tensor(senal):
        senal = tf.squeeze(senal)
        senal = tensor_to_nparray(senal)
    else:
        senal = np.squeeze(senal)
        
    # Plotear Figura
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    fig.patch.set_facecolor('white')
    specshow(senal, y_axis=y_axis, fmin=0, fmax=sampling_freq/2, x_axis='time', sr=sampling_freq, hop_length=hop_length, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel('Tiempo [s]')
    if y_axis != None:
        ax.set_ylabel('Frecuencia [Hz]')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    # Guardar Figura
    if save_plot == True:
        name = title
        fmt = 'png'
        file_name = f'{title}.{fmt}'
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0, dpi=300)



def plot_3spectrogram(clean, noisy, pred, sampling_freq, hop_length, width=5, height=5, y_axis=None, cmap='inferno', save_plot=False):
    '''
    Esta función grafica el espectrograma de la señal de audio.
    
    Parámetros:
    - 
    Returns:
    - N/A
    '''
    if tf.is_tensor(clean):
        clean = tf.squeeze(clean)
        clean = tensor_to_nparray(clean)
    else:
        clean = np.squeeze(clean)
        
    if tf.is_tensor(noisy):
        noisy = tf.squeeze(noisy)
        noisy = tensor_to_nparray(noisy)
    else:
        noisy = np.squeeze(noisy)
        
    if tf.is_tensor(pred):
        pred = tf.squeeze(pred)
        pred = tensor_to_nparray(pred)
    else:
        pred = np.squeeze(pred)
        
    # Plotear Figura
    fig, ax = plt.subplots(1, 3, figsize=(width, height))
    fig.patch.set_facecolor('white')
    
    specshow(noisy, y_axis=y_axis, fmin=0, fmax=sampling_freq/2, x_axis='time', sr=sampling_freq, hop_length=hop_length, cmap=cmap, ax=ax[0])
    ax[0].set_title('Audio con ruido')
    ax[0].set_xlabel('Tiempo [s]')
    if y_axis != None:
        ax[0].set_ylabel('Frecuencia [Hz]')
        
    specshow(pred, y_axis=y_axis, fmin=0, fmax=sampling_freq/2, x_axis='time', sr=sampling_freq, hop_length=hop_length, cmap=cmap, ax=ax[1])
    ax[1].set_title('Predicción de audio optimizado')
    ax[1].set_xlabel('Tiempo [s]')
    ax[1].get_yaxis().set_visible(False)
    if y_axis != None:
        ax[1].set_ylabel('Frecuencia [Hz]')        
        
    im = specshow(clean, y_axis=y_axis, fmin=0, fmax=sampling_freq/2, x_axis='time', sr=sampling_freq, hop_length=hop_length, cmap=cmap, ax=ax[2])
    ax[2].set_title('Audio limpio')
    ax[2].set_xlabel('Tiempo [s]')
    ax[2].get_yaxis().set_visible(False)
    if y_axis != None:
        ax[2].set_ylabel('Frecuencia [Hz]')   

    fig.colorbar(im, ax=ax[2], format='%+2.0f dB')
    plt.tight_layout()
    
    # Guardar Figura
    if save_plot == True:
        title = 'Plot_3spectrogram graph'
        name = title
        fmt = 'png'
        file_name = f'{title}.{fmt}'
        plt.savefig(file_name, bbox_inches='tight', pad_inches = 0, dpi=300)


def plot_spectrogram_plt(senal, width=5, height=5, show_axis=True, title='Espectrograma.'):
    '''
    Esta función grafica el espectrograma de la señal de audio en escala de grises.
    
    Inputs:
    - senal: Señal de audio a graficar.
    - fs: Frecuencia de muestreo del audio.
    - hop_length: Nro. muestras que se va a mover la ventana en cada desplazamiento (hop: salto)
    - width: Ancho de la grafica.
    - height: ALtura de la grafica.
    - show_axis: Mostrar información en los ejes de la grafica.
    Outputs:
    - N/A
    '''    
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    plt.imshow(senal, cmap='inferno', aspect='auto')
    ax.invert_yaxis()
    if show_axis == True:
        fig.patch.set_facecolor('white')
        ax.set_title(title)
        ax.set_xlabel('Ventanas')
        ax.set_ylabel('Divisiones en Frecuencia')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
    else:
        plt.axis("off")   # turns off axes
        plt.axis("tight")  # gets rid of white border


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    # Grafica de Función de Perdida vs Epocas
    fig, ax=plt.subplots(1,1,figsize=(5,5))
    fig.patch.set_facecolor('white')

    for l in loss_list:
        ax.plot(epochs, history.history[l], 'cornflowerblue', label='Training loss')
    for l in val_loss_list:
        ax.plot(epochs, history.history[l], 'orange', label='Validation loss')
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.legend(loc=1)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show;



def load_mono_wav(path, sampling_freq, resampling_freq=False, verbose=False):
    # Load encoded wav file
    file_contents = tf.io.read_file(path)
    # Decode wav (tensors by channels) 
    audio, _ = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    audio = tf.squeeze(audio, axis=-1)
    
    if resampling_freq != False:
        audio = tfio.audio.resample(audio, rate_in=tf.get_static_value(sampling_freq), rate_out=tf.get_static_value(resampling_freq))
        sampling_freq = resampling_freq
    
    if verbose == True:
        print(f'Frecuencia de muestreo: {sampling_freq} Hz')
    
    return audio, sampling_freq


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10.0, dtype=numerator.dtype))
    log10x = numerator / denominator
    return log10x


def magphase(D):
    mag = tf.math.abs(D)
    D_real = tf.math.real(D)
    D_imag = tf.math.imag(D)
    
    phase = tf.where(tf.equal(mag, 0),
                     # If mag == 0:
                     tf.dtypes.complex(tf.cast(1, tf.float32),
                                       tf.cast(0, tf.float32)),
                     # else:
                     tf.dtypes.complex((D_real / mag),
                                       (D_imag / mag))
                    )
    
    mag = tf.cast(mag, tf.float32)
    phase = tf.cast(phase, tf.complex64)
    
    return mag, phase


def amplitude_to_db_numpy(S, top_db=TOP_DB, amin=1e-10):
    magnitude1 = np.abs(S)
    ref_value1 = np.max(magnitude1)
    
    power1 = np.square(magnitude1)
    ref_value1 = ref_value1**2
    amin1 = amin**2
    
    S_db1 = 10.0 * np.log10( np.maximum(amin1, power1) )
    S_db1 -= 10.0 * np.log10( np.maximum(amin1, ref_value1) )
    S_db1 = np.maximum(S_db1, S_db1.max() - top_db)
    return S_db1


def amplitude_to_db(S, top_db=TOP_DB, amin=1e-10):    
    # Si la entrada dada por el usuario es compleja, se toma el valor absoluto
    # y se descarta la parte imaginaria
    magnitude = tf.math.abs(S)
    # Se toma el valor de magnitud maximo como referencia
    ref_value = tf.reduce_max(magnitude)
    
    # Se convierte el espectrograma a Potencia
    power = tf.math.pow(magnitude, 2)
    ref_value = tf.math.pow(ref_value, 2)
    amin = tf.math.pow(amin, 2)
    
    # Se convierte a escala logaritmica (Decibeles)
    S_db = 10.0 * log10( tf.math.maximum(amin, power) )
    S_db -= 10.0 * log10( tf.math.maximum(amin, ref_value) )
    S_db = tf.math.maximum(S_db, tf.reduce_max(S_db) - top_db)

    return S_db


def get_stft_log_magnitude_librosa(audio, n_fft, hop_length, win_length, window, sampling_freq, verbose=False):
    '''
    Esta función calcula la STFT en escala logaritmica de la señal de entrada de acuerdo a los argumentos dados.
    
    Inputs:
    - senal: Señal de entrada
    - fs: Frecuencia de Muestreo
    - n_ftt: Nro. de muestras para cada ventana donde se va a realizar la stft. Recomendación: Potencia de dos.
    - hop_length: Nro. muestras que se va a mover la ventana en cada desplazamiento (hop: salto)
    - window: Tipo de ventana.
    - verbose: Activar o desactivar el debugging de la STFT.
    
    Outputs:
    - stft_mag_db: Magnitud en escala logaritmica de la Transformada de Fourier.
    - stft_phase: Fase de la Transformada de Fourier.
    '''
    sampling_freq = tf.get_static_value(sampling_freq)

    # Se calcula la Transformada de Fourier de Tiempo Corto
    stft = lb.stft(tensor_to_nparray(audio),
                   n_fft=n_fft,
                   hop_length=hop_length,
                   win_length=win_length,
                   window=window,
                   center=True)

    # Se separa el espectrograma D en sus componentes de magnitud S y fase P, tal que D=S*P
    stft_mag_lineal, stft_phase = lb.magphase(stft)

    # Se convierte el espectrograma de amplitud lineal a un espectrograma escalado en dB.
    stft_mag_db = lb.amplitude_to_db(stft_mag_lineal, ref=np.max)

    # Conversión de los arrays a tensores de TensorFlow
    stft_mag_db = tf.convert_to_tensor(stft_mag_db, dtype=tf.float32)
    stft_phase = tf.convert_to_tensor(stft_phase, dtype=tf.complex64)
    
    # Se aumenta la dimensión para que el algoritmo lo tome como imagen de 1 canal
    stft_mag_db = tf.expand_dims(stft_mag_db, -1)
    stft_phase = tf.expand_dims(stft_phase, -1)
        
    # Datos de la transformada
    if verbose == True:
        print(f'Nro. de muestras dentro de cada ventana: {n_fft}')
        print(f'Solapamiento entre ventanas: {(hop_length/n_fft)*100} %')
        print(f'Duración de cada ventana: {(n_fft/sampling_freq) * 1000:.3f} ms')
        print(f'Dimensiones de la Mátriz STFT resultante: {stft.shape}')
        print(f'Resolución en frecuencia del espectograma: {stft.shape[0]} divisiones.')
        print(f'Resolución en tiempo / Número de ventanas a lo largo de la serie de tiempo: {stft.shape[1]} ventanas.')
            
    return stft_mag_db, stft_phase


def get_stft_log_magnitude(audio, n_fft, hop_length, win_length, window, sampling_freq, verbose=False):
    '''
    Esta función calcula la STFT en escala logaritmica de la señal de entrada de acuerdo a los argumentos dados.
    
    Inputs:
    - senal: Señal de entrada
    - fs: Frecuencia de Muestreo
    - n_ftt: Nro. de muestras para cada ventana donde se va a realizar la stft. Recomendación: Potencia de dos.
    - hop_length: Nro. muestras que se va a mover la ventana en cada desplazamiento (hop: salto)
    - window: Tipo de ventana.
    - verbose: Activar o desactivar el debugging de la STFT.
    
    Outputs:
    - stft_mag_db: Magnitud en escala logaritmica de la Transformada de Fourier.
    - stft_phase: Fase de la Transformada de Fourier.
    '''    
    if window == 'hann':
        window = tf.signal.hann_window
        
    stft = tf.signal.stft(audio,
                          fft_length=n_fft,
                          frame_step=hop_length,
                          frame_length=win_length,
                          window_fn=window,
                          pad_end=True,
                          )
    
    # Se separa el espectrograma D en sus componentes de magnitud S y fase P, tal que D=S*P
    stft_mag_lineal, stft_phase = magphase(stft)
    
    # Se convierte el espectrograma de amplitud lineal a un espectrograma escalado en dB.
    stft_mag_db = amplitude_to_db(stft_mag_lineal)
    
    # Se aumenta la dimensión para que el algoritmo lo tome como imagen de 1 canal
    stft_mag_db = tf.expand_dims(stft_mag_db, -1)
    stft_phase = tf.expand_dims(stft_phase, -1)

    stft_mag_db = tf.image.transpose(stft_mag_db)
    stft_phase = tf.image.transpose(stft_phase)
        
    # Datos de la transformada
    if verbose == True:
        print(f'Nro. de muestras dentro de cada ventana: {n_fft}')
        print(f'Solapamiento entre ventanas: {(hop_length/n_fft)*100} %')
        print(f'Duración de cada ventana: {(n_fft/sampling_freq) * 1000:.3f} ms')
        print(f'Dimensiones de la Mátriz STFT resultante: {stft.shape}')
        print(f'Resolución en frecuencia del espectograma: {stft.shape[0]} divisiones.')
        print(f'Resolución en tiempo / Número de ventanas a lo largo de la serie de tiempo: {stft.shape[1]} ventanas.')
            
    return stft_mag_db, stft_phase


def db_to_amplitude(S_db, ref=1.0):
    S_linear = tf.math.pow(ref, 2) * tf.math.pow(10.0, 0.1 * S_db)
    S_linear = tf.math.pow(S_linear, 0.5)

    return S_linear


def get_istft_log_magnitude(stft_mag_db, stft_phase, n_fft, hop_length, win_length, window, sampling_freq, verbose=False):
    '''
    Esta función calcula la ISTFT del espectrograma de entrada de acuerdo a los argumentos dados.
    
    Inputs:
    - stft_mag_db: Magnitud en escala de decibeles del espectrograma de entrada
    - stft_phase: Fase del espectrograma de entrada
    - n_ftt: Nro. de muestras para cada ventana donde se va a realizar la stft. Recomendación: Potencia de dos.
    - hop_length: Nro. muestras que se va a mover la ventana en cada desplazamiento (hop: salto)
    - window: Tipo de ventana.
    - verbose: Activar o desactivar el debugging de la STFT.
    Outputs:
    - istft: Transformada Inversa de Fourier de Tiempo Corto de la señal.
    '''
    # Remueve la dimension correspondiente al numero de canales de la imagen
    stft_mag_db = tf.squeeze(stft_mag_db)
    stft_phase = tf.squeeze(stft_phase)

    # Se convierte el espectrograma escalado en dB a un espectrograma de amplitud lineal
    stft_mag_lineal = db_to_amplitude(stft_mag_db, ref=1.0)    
    stft_mag_lineal = tf.cast(stft_mag_lineal, tf.complex64)
    
    # Se une el espectrograma D a partir de sus componentes de magnitud S y fase P, tal que D=S*P
    audio_reverse_stft = tf.math.multiply(stft_mag_lineal, stft_phase)
    

    # Se calcula la Transformada Inversa de Fourier de Tiempo Corto
    istft = lb.istft(tensor_to_nparray(audio_reverse_stft),
                     n_fft=n_fft,
                     hop_length=hop_length,
                     win_length=win_length,
                     window=window,
                     center=True)
    
    # Conversión de los arrays a tensores de TensorFlow
    istft = tf.convert_to_tensor(istft, dtype=tf.float32)

    # Datos de la transformada
    if verbose == True:
        print(f'Nro. de muestras dentro de cada ventana: {n_fft}')
        print(f'Solapamiento entre ventanas: {(hop_length/n_fft)*100} %')
        print(f'Dimensiones de la Mátriz ISTFT resultante: {istft.shape}')
        print(f'Muestras en el audio: {istft.shape[0]}.')
            
    return istft


def listen(audio, sampling_freq):
    sampling_freq = tf.get_static_value(sampling_freq)

    if tf.is_tensor(audio):
        audio = tensor_to_nparray(audio)
    
    return ipd.Audio(audio, rate=sampling_freq)


def resize_spectrogram(stft_mag, stft_phase, input_width, input_height, output_width, output_height, channels):
    stft_mag.set_shape([input_width, input_height, channels])
    
    stft_mag_resized = tf.image.resize(stft_mag, [output_width, output_height], antialias=True)
    
    return stft_mag_resized, stft_phase


def normalize_spectrogram(stft_mag, stft_phase, top_db):
    stft_mag_normalized = stft_mag / top_db
    return stft_mag_normalized, stft_phase


def denormalize_spectrogram(stft_mag, stft_phase, top_db):
    stft_mag_denormalized = stft_mag * top_db
    return stft_mag_denormalized, stft_phase

def preprocess(file_path, sampling_freq_original, resampling_freq, max_dur, n_fft, hop_length, win_length, window,
               input_width, input_height, rescale_width, rescale_height, channels, top_db, verbose=False):
    
    # Load and downsample
    audio, sampling_freq = load_mono_wav(file_path, sampling_freq_original, resampling_freq, verbose)

    audio = pad_trunc_audio(audio, sampling_freq, max_dur)
    
    stft_mag, stft_phase = get_stft_log_magnitude(audio,
                                                  n_fft,
                                                  hop_length,
                                                  win_length,
                                                  window,
                                                  sampling_freq,
                                                  verbose,
                                               )
    
    stft_mag, stft_phase = resize_spectrogram(stft_mag,
                                              stft_phase,
                                              input_width,
                                              input_height,
                                              rescale_width,
                                              rescale_height,
                                              channels
                                             )
    
    stft_mag, stft_phase = normalize_spectrogram(stft_mag, stft_phase, top_db)
        
    if verbose == True:
        print(f'Dimensiones originales del espectrograma: {stft_phase.shape}')
        print(f'Dimensiones de salida del espectrograma: {stft_mag.shape}')
    
    return stft_mag, stft_phase


def postprocess(stft_mag, stft_phase, input_width=RESCALE_WIDTH, input_height=RESCALE_HEIGHT, rescale_width=INPUT_WIDTH,
                rescale_height=INPUT_HEIGHT,channels=CHANNELS,top_db=TOP_DB):

    stft_mag, stft_phase = resize_spectrogram(stft_mag,
                                              stft_phase,
                                              input_width,
                                              input_height,
                                              rescale_width,
                                              rescale_height,
                                              channels
                                             )
    
    stft_mag, stft_phase = denormalize_spectrogram(stft_mag, stft_phase, top_db)
        
    return stft_mag, stft_phase

def UNETencoder(inputs, n_filters, kernel_size, activation_layer, kernel_init, batch_normalization, dropout, max_pooling):
    """
    Esta función utiliza múltiples capas de convolución, max pool, activación relu para crear una arquitectura de aprendizaje.
    Una inicialización adecuada evita el problema de la explosión y el desvanecimiento de los gradientes.
    Se puede añadir un dropout para regularizar y evitar el sobreajuste. 
    Parámetros:
    - inputs: Entrada al encoder
    - n_filters: Número de Filtros
    - kernel_size: Tamaño del Kernel
    - activation_layer: Capa de Activación
    - kernel_init: Inicialización de Kernel
    - dropout: Dropout
    - max_pooling: Opción booleana para realizar max pooling
    Returns:
    - next_layer: Valores de activación para la siguiente capa.
    - skip_connection: Conexión de salto que se utilizará en el decodificador.
    """
    if activation_layer == 'leakyrelu':
        activation = None
    else:
        activation = activation_layer
    
    # Primera Capa de Convolución
    conv = Conv2D(n_filters, 
                  kernel_size,
                  activation=activation,
                  padding='same', # Para asegurar que el tamaño de la imagen no disminuya.
                  kernel_initializer=kernel_init)(inputs)
    
    if activation_layer == 'leakyrelu':
        conv = LeakyReLU()(conv)
    
    # Segunda Capa de Convolución
    conv = Conv2D(n_filters, 
                  kernel_size,
                  activation=activation,
                  padding='same', # Para asegurar que el tamaño de la imagen no disminuya.
                  kernel_initializer=kernel_init)(conv)
    
    if activation_layer == 'leakyrelu':
        conv = LeakyReLU()(conv)
    
    # Normalización del lote. Se normalizará la salida de la última capa basándose en la media y la desviación estándar del lote
    if batch_normalization == True:
        conv = BatchNormalization()(conv, training=False)

    # En caso de sobreajuste, el dropout regularizará la pérdida y el cálculo del gradiente, para reducir la influencia de los pesos en la salida
    if dropout > 0:     
        conv = Dropout(dropout)(conv)

    # El pooling reduce el tamaño de la imagen manteniendo el mismo número de canales
    # El pooling se ha mantenido como opcional ya que la última capa del codificador no utiliza el pooling.
    # Se utiliza un stride de 2 para recorrer la imagen de entrada, y considera el máximo del slice de entrada para el cálculo de la salida.
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # Las Skip Conecction se introducirán en el decoder para evitar la pérdida de información durante las convoluciones transpuestas.
    skip_connection = conv
    
    return next_layer, skip_connection


def UNETdecoder(prev_layer_input, skip_layer_input, n_filters, kernel_size, activation_layer, kernel_init, dropout):
    """
    El bloque decodificador utiliza primero la convolución de transposición para escalar la imagen a un tamaño mayor
    y luego fusiona el resultado con los resultados de la capa de salto del bloque codificador
    La adición de 2 convoluciones con el mismo relleno ayuda a aumentar la profundidad de la red para mejorar las predicciones
    La función devuelve la salida de la capa decodificada
    Parámetros:
    - prev_layer_input: 
    - skip_layer_input:
    - n_filters: Número de Filtros
    - kernel_size: Tamaño del Kernel
    - activation_layer: Capa de Activación
    - kernel_init: Inicialización de Kernel
    Returns:
    - conv: 
    """
    
    if activation_layer == 'leakyrelu':
        activation = None
    else:
        activation = activation_layer
        
    # Capa de convolución transpuesta para aumentar el tamaño de la imagen
    up = Conv2DTranspose(n_filters,
                         (kernel_size, kernel_size),
                         strides=(2,2),
                         padding='same')(prev_layer_input)

    # Se combina la skip connection del bloque anterior para evitar la pérdida de información
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Primera Capa de Convolución
    conv = Conv2D(n_filters,
                  kernel_size,
                  activation=activation,
                  padding='same', # Para asegurar que el tamaño de la imagen no disminuya.
                  kernel_initializer=kernel_init)(merge)

    if activation_layer == 'leakyrelu':
        conv = LeakyReLU()(conv)
        
    # Segunda Capa de Convolución
    conv = Conv2D(n_filters,
                  kernel_size,
                  activation=activation,
                  padding='same', # Para asegurar que el tamaño de la imagen no disminuya.
                  kernel_initializer=kernel_init)(conv)
    
    if activation_layer == 'leakyrelu':
        conv = LeakyReLU()(conv)
        
    # En caso de sobreajuste, el dropout regularizará la pérdida y el cálculo del gradiente, para reducir la influencia de los pesos en la salida
    if dropout > 0:     
        conv = Dropout(dropout)(conv)
        
    return conv