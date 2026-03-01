import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_dst_net(window_size, num_features, num_classes):
    """
    Builds the Deep Spatiotemporal Synergy Network (DST-Net).
    Architecture: 1D-CNN (Local Morphology) -> LSTM (Global Sequence) -> Dense (Classifier)
    """
    model = Sequential(name="DST-Net")
    
    # Spatial Path: 1D-CNN Module for Local Gradients
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
                     input_shape=(window_size, num_features), name="Conv1D_1"))
    model.add(MaxPooling1D(pool_size=2, name="MaxPool_1"))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name="Conv1D_2"))
    
    # Temporal Path: LSTM Module for Sedimentary Context
    model.add(LSTM(units=128, return_sequences=False, name="LSTM_Module"))
    
    # Synergy Fusion & Output Layer
    model.add(Dropout(0.3, name="Dropout"))
    model.add(Dense(units=num_classes, activation='softmax', name="Softmax_Output"))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model