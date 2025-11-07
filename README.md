# NEURONAL NETWORK

DESCRIPCIÓN:

    El objetivo que se espera lograr con esta primera práctica de la asignatura Fundamentos de los Sistemas Inteligentes es la comparación de redes neurales con distintas características. Para ello se emplean funciones de la librería PyTorch que nos permiten crear capas de neuronas, así como utilizar facilmente funciones de activación (RELU, Sigmoid), funciones de perdida (MSELoss, CrossEntropyLoss) y optimizadores (SGD, Adam).

DATASET ELEGIDO:

    Con el propósito de entrenar las redes para la detección de señales de tráfico hemos utilizado este dataset que contiene imágenes de 15 clases distintas de semáforos y señales:

        https://www.kaggle.com/datasets/pkdarabi/cardetection
    
    Contamos con 3 directorios (train, test, valid) y cada uno de ellos tiene una carpeta para las imágenes y otra para las etiquetas. El tamaño de las imágenes es de 416x416 píxeles.

TIPOS DE REDES NEURONALES:

    - Simple Network

        Para el primer diseño hemos creado una red neuronal con dos capas ocultas (512 y 128 neuronas), y una capa de salida de 15 clases. Como función de activación hemos usado la sigmoide puesto que fue la que vimos en primer lugar en clase de teoría, y en la capa de salida aplicamos softmax. 

CANCIÓN:

    A modo de resumen hemos escrito esta letra y generado una canción sobre la práctica con la ayuda de Suno AI:

        https://drive.google.com/file/d/1I_AofIVKEJFz1Ctkm6tKBoN2TtxnousO/view?usp=sharing

        Me voy a descargar este dataset de kaggle,
        que tiene imagenes de 15 clases de señales.
        Yolo, Yolo, Yolo.
        Y cuando voy a ver que hay en cada carpeta,
        me fijo en el formato que tienen las etiquetas.
        Yolo, Yolo, Yolo.

        Uoooooooooo
        que lento entrena la red neuronal,
        Si poooooooongo
        un valor bajo de learning rate.

        Añado luego alguna capa convolucional,
        para lograr la precisión que me haga aprobar.
        Yolo, Yolo, Yolo.
        Y como aún me está saliendo un porcentaje bajo,
        Intento usar ahora un modelo pre entrenado.
        Yolo, Yolo, Yolo.

        Uoooooooooo
        que lento entrena la red neuronal,
        Si poooooooongo
        un valor bajo de learning rate.
