import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------

class FraudPreprocessor:

    def __init__(self):
        self.pipeline = None
        self.feature_names_in_ = None

    def preprocess(self, df, target_col=None, training=False):

        df = df.copy()

        if training and target_col:
            y = df[target_col].values
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        if training:
            X_processed = self.pipeline.fit_transform(X)
        else:
            X_processed = self.pipeline.transform(X)

        return X_processed, y


# ---------------------------------------------------
# GRU MODEL
# ---------------------------------------------------

def create_model(input_dim):

    inputs = Input(shape=(input_dim, 1))

    x = GRU(32)(inputs)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model