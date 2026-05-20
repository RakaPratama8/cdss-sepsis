import tensorflow as tf

def build_sepsis_gru(input_shape):
    model = tf.keras.Sequential([
        # Using Input(shape) instead of input_shape in the first layer to avoid warnings
        tf.keras.Input(shape=input_shape, name="Input_Layer"),

        tf.keras.layers.Masking(mask_value=0.0, name="Padding_Mask"),

        tf.keras.layers.GRU(
            64,
            return_sequences=False,
            dropout=0.2,
            name="GRU_Core"
        ),

        tf.keras.layers.Dense(32, activation='relu', name="Dense_Interpreter"),
        tf.keras.layers.Dropout(0.3, name="Regularization_Drop"),
        tf.keras.layers.Dense(1, activation='sigmoid', name="Sepsis_Probability")
    ])

    metrics = [
        tf.keras.metrics.AUC(name='roc_auc'),
        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model