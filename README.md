# Signals analysis

Este es el trabajo final de la materia Análisis de datos científicos
dictada por Rodrigo Ramele (ITBA).

El objetivo es analizar la señal proveniente de un encefalograma con el
objetivo de aplicar tecnicas supervisadas o no supervisadas de
aprendizaje automático. Los datos tienen una duración de 12 minutos
aproximadamente y contienen siguiendo la siguiente secuencia de
acciones:

    |---- BASELINE --------|
    |---- TOSER ------|
    |---- RESPIRAR FONDO ------- |
    |---- RESPIRAR RAPIDO ----|
    |---- CUENTA MENTAL --------|
    |---- COLORES VIOLETA ------|
    |---- COLORES ROJO --------|
    |---- SONREIR -----|
    |---- DESEGRADABLE -----| 
    |---- AGRADABLE --------|
    |---- PESTANEOS CODIGO ------ |

Baseline corresponde a una situación normal del individuo y el resto de
las acciones se realizan de forma secuencial. Un video acompaña los
datos mostrando la secuencia y el timestamp correspondiente.

En este trabajo se realiza un análisis exploratorio y preparación de los
datos. Luego se aplican distintas técncias de clasificación con el fin
de poder distinguir la señal de "tos" tomando como referencia el
"baseline". En una segunda parte se vuelven a aplicar las técnicas de
clasificación para distinguir entre respiración profunda y respiración
rápida.

El notebook utilizado se encuentra aqui:

https://github.com/natdebandi/signals-analysis/blob/275efc7a7ae000594724e9f25d47aa4a2593e9ac/signal-analysis.ipynb

# Análisis exploratorio

Lo primero fue levantar el dataset y realizar un análisis exploratorio.

![señal completa](https://github.com/natdebandi/signals-analysis/blob/090c96eea4d62242a10480d8182eebca60105cc8/images/Signal1.png?raw=true)

A partir del video se marcaron los tiempos según el timestamp en el que
suceden los cambios. Cada tramo tiene una duracion de 60 seg
aproximadamente y está separada por pestañeos.

Para poder sincronizar y marcar los cambios en la señal con un label se
analizó la diferencia temporal: el primer valor de la serie es
2022-06-22 19:06 mientras que el video arranca en el 2022-06-22 20:05.
Se creó una nueva columna DATE con el timestamp sincronizado de modo e
poder seguir más facilmente las ocurencias en la señal.

    signals['date']=signals['date'] + timedelta(minutes=59)

Luego se separó agregó una etiqueta con los tiempos marcados y se
grafico usando seaborn.

![Señal con la identificación de cada cambio](https://github.com/natdebandi/signals-analysis/blob/a941976b108a3575c28488056a6f889ef29000e1/images/signal_labels.png?raw=true)


# Separación de señales y extracción de features

Se tomaron por separado la señal baseline y la señal de tos. En cada una
tome 50 segundos centrales para evitar el "ruido" de los pestañeos y el
cambio de secuencias.

    signals_baseline=signals[(signals.date >'2022-06-22 20:06:25') & (signals.date <'2022-06-22 20:07:15')]
    signals_tos=signals[(signals.date >'2022-06-22 20:07:25') & (signals.date <'2022-06-22 20:08:15')]

Se le aplicó a cada señal un filtro temporal de convoolución y se probó
elminando el baseline de la señal(se uso getbaseline).

    #aplico la funcion getbaseline y la de eliminar ruido por medio de convolución
    data=signals_baseline.values
    eeg_val=data[:,2]
    eeg_val=eeg_val-getBaseline(eeg_val)

    #filtro convolución
    windowlength = 10
    eeg_val = np.convolve(eeg_val, np.ones((windowlength,))/windowlength, mode='same')
    print(eeg_val)
    signals_baseline['eeg_val']=eeg_val

A partir de ahi se extrae de cada una de las señales: "signals baseline"
y "signal tos". Se observa que se suaviza la señal levemente,
destacándose un poco más los picos de cada fragmento.

![fragmentos de señal baseline y tos](https://github.com/natdebandi/signals-analysis/blob/a941976b108a3575c28488056a6f889ef29000e1/images/signals_bt.png?raw=true)

Para extraer los features utilicé las funciones que estaban en el
proyecto (crest_factor, hjorth y pfd). Cree también una función que dado
un vector de numpy devuelve los features aplicando estas funciones.
Luego otra función que aplicará esta para extraer a la señal entera
todos los features usando una determinada ventana.

    ##creo una funcion para obtener los features a partir de un vector de nunpy de frecuencia
    def getfeatures(eeg_in):
        features=[]
        ptp = abs(np.max(eeg_in)) + abs(np.min(eeg_in))
        features.insert(1,ptp)

        rms = np.sqrt(np.mean(eeg_in**2))
        features.insert(1,rms)

        cf = crest_factor(eeg_in)
        features.insert(1,cf)

        entropy = stats.entropy(list(Counter(eeg_in).values()), base=2)
        features.insert(1,entropy)

        activity, complexity, morbidity = hjorth(eeg_in)
        features.insert(1,activity)
        features.insert(1,complexity)
        features.insert(1,morbidity)
        fractal = pfd(eeg_in)
        features.insert(1,fractal)
        return features

    def features_signals(signal,ventana):
        i=0
        features=[]
        N_signals=signal.timestamp.count()
        #print(N_signals)
        while i<N_signals:
         #   print(i)
            s_temp = signal.iloc[i:i+ventana,:]
            #print(s_temp['date'])
            data = s_temp.values
            eeg = data[:,7]
            f_temp=getfeatures(eeg)
          #  print(f_temp)
            features.append(f_temp)
            i=i+ventana
        return(features)

# Clasificación: tos y baseline

Se calcularon los features haciendo distintas pruebas, usando ventanas
de 1,2 y 3 segundos multiplicadas por la frecuencia de sampleo 512).Se
decidió tomar la ventana de 1 segundo para tener más datos.

![Scatterplot del feature entropia vs activity](https://github.com/natdebandi/signals-analysis/blob/7df824f685a74a89705b8e516e8da759edb1e640/images/features1.png?raw=true)


Se hicieron distintas pruebas de clasificación. Se reutilizó el código
existente en el proyecto (signalfeatureclassification.py y
onepassclassifier.py).

Se aplicó SVM, KNeiborhoods (KNN), regresión logística, analisis
discriminante (LDS) y árbol de decisión (DecTree). Los resultados fueron
bastante similares con todas las técnicas donde no pudo discriminarse
satisfactoriamente la diferencia entre ambas señales El mejor resultado
se obtuvo con LDA pero no es tampoco satisfactorio.

    Feature 1 Size 51,8
    Feature 2 Size 51,8
    Boundary 51:
    Training Dataset Size 51,8
    Test Dataset Size 50,8
    Trivial: ROC AUC=0.500
    SVM: ROC AUC=0.472
    kNN: ROC AUC=0.498
    LogReg: ROC AUC=0.568
    LDA: ROC AUC=0.668
    Decision Tree: ROC AUC=0.493
    Random Forest: ROC AUC=0.549

![Curvas ROC de las técnicas de clasificación (baseline vs tos)](https://github.com/natdebandi/signals-analysis/blob/7df824f685a74a89705b8e516e8da759edb1e640/images/clasificacion1.png?raw=true)


También se aplicó una red neuronal simple (keras) la cual obtuvo
resultados similares a las otras técnicas.

    Keras Model Accuracy: 0.540000
                  precision    recall  f1-score   support

        baseline       0.64      0.27      0.38        26
             Tos       0.51      0.83      0.63        24

        accuracy                           0.54        50
       macro avg       0.57      0.55      0.51        50
    weighted avg       0.58      0.54      0.50        50

# Clasificación: respiración profunda y rapida

A modo de prueba se realizó también una clasificación similar entre la
señal de respiración profunda y rápida. En este caso las señales se
tomaron con 60 segundos y no se aplicaron técnicas de filtrado ni
elimnación del baseline.

Se aplicaron las mismas estrategias de análisis obteniendo los features
a partir de una venta de 2 segundos (512 de frecuencia de sampleo). En
este caso los resultados fueron mucho mejores:

![Curvas ROC de las técnicas de clasificación (respiración fuerte vs rápida)](https://github.com/natdebandi/signals-analysis/blob/7df824f685a74a89705b8e516e8da759edb1e640/images/clasificacion2.png?raw=true)


    Trivial: ROC AUC=0.500
    SVM: ROC AUC=0.871
    kNN: ROC AUC=0.877
    LogReg: ROC AUC=0.924
    LDA: ROC AUC=0.866
    Decision Tree: ROC AUC=0.732
    Random Forest: ROC AUC=0.795

En este caso el mejor resultado se obtuvo con la regresión logística
(0.94) aun cuando no se optimizaron los parámetros de nigún
clasificador. Se realizó asimismo la prueba con la misma red neuronal
simple y se obtuvo un mejor resultado que en el caso anteror pero no tan
bueno como la regresión logística:

    Keras Model Accuracy: 0.766667
                  precision    recall  f1-score   support

        baseline       0.64      0.70      0.67        10
             Tos       0.84      0.80      0.82        20

        accuracy                           0.77        30
       macro avg       0.74      0.75      0.74        30
    weighted avg       0.77      0.77      0.77        30

# Conclusiones

Los resultados que se presentaron son de caracter exploratorio con el
objetivo de aprender el tratamiento y manejo de señales y posibles
formas de clasificación. Se observa que efectivamente se pueden
distinguir y clasificar eventos dentro de señales a partir de técnicas
de aprendizaje supervisado, aunque seguramente se requiera de numerosos
ajustes para poder obtener resultados efectivos.
