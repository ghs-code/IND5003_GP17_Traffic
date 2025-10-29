import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks # type: ignore
from sklearn.metrics import accuracy_score
from math import sqrt 


'''Train LSTM classification models

    This script trains Long Short-Term Memory (LSTM) classification models 
    for traffic congestion clustering tasks. It is designed to handle 
    preprocessed datasets containing temporal features and cluster labels 
    (1–4 corresponding to Very Light, Light, Moderate, Severe).
Workflow:
    1. Load .npz dataset files for each segment.
    2. Build and compile the LSTM classifier.
    3. Train the model with early stopping and checkpoint callbacks.
    4. Evaluate performance on the validation set and save logs in JSON format.
Input:
    - ./scripts/artifacts/phase_cluster/*.npz  
      Each file must contain:
        X_train, y_train, X_val, y_val arrays.
Output:
    - Saved model weights (.h5)
    - Training logs (.json) including accuracy metrics.

'''

PHASE_DIR = "./scripts/artifacts/phase_cluster"  
MODEL_DIR = "./scripts/artifacts/model_cluster"
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = {
    "causeway":    os.path.join(PHASE_DIR, "phase_cluster_causeway_dataset.npz"),
    "second_link": os.path.join(PHASE_DIR, "phase_cluster_second_link_dataset.npz"),
}

MODEL_OUT = {
    "causeway":    os.path.join(MODEL_DIR, "causeway_lstm_best.h5"),
    "second_link": os.path.join(MODEL_DIR, "second_link_lstm_best.h5"),
}



def build_lstm_classifier(t_in, n_features, t_out, n_classes=4):
    """
    Input: Past t_in steps, n_features per step
    Output: Future t_out steps, probability distribution of n_classes per step
          => (batch, t_out, n_classes)

    LSTM --> Dense --> Dense(t_out * n_classes) --> Reshape(t_out, n_classes) --> Softmax stepwise classification
    """

    inp = layers.Input(shape=(t_in, n_features))

    x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.Dense(32, activation="relu")(x)

    x = layers.Dense(t_out * n_classes, activation=None)(x)

 
    x = layers.Reshape((t_out, n_classes))(x)

    # softmax probability distribution by category
    out = layers.Softmax(axis=-1)(x)

    model = models.Model(inputs=inp, outputs=out)



    # - y_true Shape: (batch, t_out)
    # - y_pred Shape: (batch, t_out, n_classes)
    # - y_true is int (0..n_classes-1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    return model


def train_for_segment(seg_name, dataset_path, model_out_path,
                      max_epochs=100, batch_size=32, patience=10):

    print(f"\n=== Training segment: {seg_name} ===")
    print(f"Loading dataset from {dataset_path}")

    # 1. load data
    data = np.load(dataset_path)
    X_train = data["X_train"]  # shape: (N, T_IN, 5)
    y_train = data["y_train"]  # shape: (N, T_OUT) with values in {1,2,3,4}
    X_val   = data["X_val"]
    y_val   = data["y_val"]

    print(f"{seg_name} X_train shape: {X_train.shape}")
    print(f"{seg_name} y_train shape: {y_train.shape}")
    print(f"{seg_name} X_val shape:   {X_val.shape}")
    print(f"{seg_name} y_val shape:   {y_val.shape}")

    # 2. convert labels from {1,2,3,4} -> {0,1,2,3} for SparseCategoricalCrossentropy
    y_train_cls = (y_train.astype(np.int64) - 1)
    y_val_cls   = (y_val.astype(np.int64) - 1)

    # 3. build model
    T_IN     = X_train.shape[1]
    N_FEAT   = X_train.shape[2]
    T_OUT    = y_train.shape[1]
    N_CLASS  = 4

    model = build_lstm_classifier(T_IN, N_FEAT, T_OUT, n_classes=N_CLASS)
    model.summary()

    # 4. callbacks
    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=model_out_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1
    )

    early_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # 5. train
    history = model.fit(
        X_train, y_train_cls,
        validation_data=(X_val, y_val_cls),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[ckpt_cb, early_cb],
        verbose=1,
        shuffle=True    
    )

    # 6. evaluation on val: accuracy
    y_pred_prob = model.predict(X_val)          
    y_pred_cls0 = np.argmax(y_pred_prob, axis=-1)  
    y_pred_cls  = y_pred_cls0 + 1                  

    # overall accuracy across all steps+samples
    acc_overall = accuracy_score(
        y_val.flatten(),    # true labels in {1..4}
        y_pred_cls.flatten()
    )

    # per-step accuracy
    acc_per_step = []
    for step in range(T_OUT):
        acc_step = accuracy_score(
            y_val[:, step],
            y_pred_cls[:, step]
        )
        acc_per_step.append(float(acc_step))

    metrics = {
        "segment": seg_name,
        "val_acc_per_step": acc_per_step,
        "val_acc_overall": float(acc_overall),
        "best_model_path": model_out_path,
    }

    print(f"{seg_name} validation acc per step: {acc_per_step}")
    print(f"{seg_name} validation acc overall: {acc_overall:.4f}")
    print(f"Best model saved to {model_out_path}")

    # 7. save training log
    log_path = os.path.join(MODEL_DIR, f"{seg_name}_train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history.history,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Training log saved to {log_path}")


if __name__ == "__main__":

    # Train Causeway
    train_for_segment(
        seg_name="causeway",
        dataset_path=DATASETS["causeway"],
        model_out_path=MODEL_OUT["causeway"],
        max_epochs=100,
        batch_size=32,
        patience=10
    )

    # Train Second Link
    train_for_segment(
        seg_name="second_link",
        dataset_path=DATASETS["second_link"],
        model_out_path=MODEL_OUT["second_link"],
        max_epochs=100,
        batch_size=32,
        patience=10
    )
