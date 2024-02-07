import tensorflow as tf
import numpy as np
import librosa as lb

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