# src/VGG_help.py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, Xception
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing import image_dataset_from_directory


# ===========================================================
# 🔧  Dataset loaders
# ===========================================================

def prepare_dataset(dataset_dir, img_size=(128, 128), batch_size=32, shuffle=True):
    return image_dataset_from_directory(
        dataset_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=shuffle
    )


def prepare_mixed_dataset(original_dir, generated_dir, img_size=(128, 128), batch_size=32, mix_ratio=0.5):
    """Mistura imagens originais e sintéticas proporcionalmente."""
    orig_ds = prepare_dataset(original_dir, img_size, batch_size, shuffle=True)
    gen_ds = prepare_dataset(generated_dir, img_size, batch_size, shuffle=True)

    n_orig = int((1 - mix_ratio) * batch_size)
    n_gen = batch_size - n_orig

    mixed_ds = tf.data.Dataset.zip((orig_ds, gen_ds)).map(
        lambda o, g: (
            tf.concat([o[0][:n_orig], g[0][:n_gen]], axis=0),
            tf.concat([o[1][:n_orig], g[1][:n_gen]], axis=0)
        )
    )
    return mixed_ds.prefetch(tf.data.AUTOTUNE)


# ===========================================================
# 🧠  Model factories
# ===========================================================


def build_densenet121(input_shape, num_classes, weights="imagenet"):
    base = DenseNet121(weights=weights, include_top=False, input_shape=input_shape)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ===========================================================
# 🧮  Training utilities
# ===========================================================

def train_and_evaluate(model, train_ds, val_ds, test_ds, class_names, epochs=10, output_dir="./Trained_Models", title="model"):
    os.makedirs(output_dir, exist_ok=True)

    # Compute class weights for imbalance
    all_labels = np.concatenate([y for x, y in train_ds], axis=0)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(all_labels), y=all_labels)
    class_weights = dict(enumerate(class_weights))

    # Callbacks
    checkpoint_path = os.path.join(output_dir, f"{title}_best.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max")
    early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights,
                        callbacks=[ckpt, early], verbose=1)

    # Evaluate
    metrics = model.evaluate(test_ds, verbose=0)
    y_true, y_pred, y_prob = [], [], []

    for X, y in test_ds:
        preds = model.predict(X)
        y_prob.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, len(class_names)),
                        np.array(y_prob), multi_class='ovr')

    metrics_dict = {
        "Accuracy": report["accuracy"],
        "Weighted F1": report["weighted avg"]["f1-score"],
        "Macro F1": report["macro avg"]["f1-score"],
        "AUROC": auc
    }

    # Save metrics
    pd.DataFrame([metrics_dict]).to_csv(os.path.join(output_dir, f"{title}_metrics.csv"), index=False)

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}_confusion.png"))
    plt.close()

    print(f"✅ Training finished for {title}")
    print(metrics_dict)
    return history, metrics_dict
