import numpy as np
import tensorflow as tf

def test_tensorflow():
    # create 2 datasets
    X = np.array(np.random.random((100, 5)))  # Matrix 100 x 10 with numbers in the range [0;1]
    Y = np.array(np.random.random((100)))  # Vector of length = 100  with numbers in the range [0;1]

    # Create a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(3, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer='Adam',
        loss='mse',
        metrics=['mean_absolute_error']
    )

    # If the error does not decrease during the specified number of epochs,
    # the learning process is interrupted and the model is initialized with weights
    # with the lowest indicator of the "monitor" parameter
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        # specifies the parameter is performed, usually it is a loss function on the validation set (val_loss)
        patience=2,  # the number of epochs after which the training will end, if the indicators do not improve
        mode='min',  # specifies in which direction the error should be improved
        restore_best_weights=True  # if the parameter is set to true, then at the end of training,
        # the model will be initialized with the weights with the lowest indicator of the "monitor"parameter
    )

    # Save the model to next loading
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='my_model',  # путь к папке, где будет сохранена модель
        monitor='val_loss',
        save_best_only=True,  # если параметр установлен в true, то сохраняется только лучшая модель
        mode='min'
    )

    # Save a log of learning, which can be see in special envirement
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='log',  # path to directory
    )

    model.fit(
        X,  # Input data set
        Y,  # set of right answers
        validation_split=0.2,  # 20% of input data set split to validation data
        epochs=50,  # the learning process will end in 50 epochs
        batch_size=8,  # dataset split to batches of 8 items each
        callbacks=[
            early_stopping,
            model_checkpoint,
            tensorboard
        ]
    )

    X_test = np.array(np.random.random((10, 5)))
    Y_test = np.array(np.random.random(10))

    res = model.evaluate(X_test, Y_test)
    print("loss and mean_absolute_error", res)

    predictions = model.predict(X_test)
    print(predictions)

    return 0