import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import shutil  # Para copy em split

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate_from_arrays(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_fn, input_shape, num_classes,
    epochs=5, batch_size=32, title="Model"
):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # Criar modelo
    model = model_fn(input_shape=input_shape, num_classes=num_classes)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
    ]

    # Treinar
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Avaliação
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Test Loss": test_loss,
        "Test Accuracy": test_acc
    }

    return model, metrics, cm

def prepare_dataset(dataset_dir, class_labels, target_size=(128, 128)):
    X = []
    y = []

    for label_index, class_label in enumerate(class_labels):
        label_folder = os.path.join(dataset_dir, class_label)
        if not os.path.exists(label_folder):
            continue
        image_files = os.listdir(label_folder)

        for img_file in image_files:
            img_path = os.path.join(label_folder, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.vgg16.preprocess_input(img)
            X.append(img)
            y.append(label_index)

    if len(X) == 0:
        # Força empty 4D para compatibilidade com vstack
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32), np.array([], dtype=np.int32)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]

def prepare_mixed_dataset(original_dir, generated_dir, class_labels, target_size=(128, 128), mix_ratio=0.5):
    X_orig, y_orig = prepare_dataset(original_dir, class_labels, target_size)
    X_gen, y_gen = prepare_dataset(generated_dir, class_labels, target_size)
    X_mixed = []
    y_mixed = []
    for label in range(len(class_labels)):
        orig_idx = np.where(y_orig == label)[0]
        gen_idx = np.where(y_gen == label)[0]
        n_orig = len(orig_idx)
        n_gen = min(int(n_orig / mix_ratio) if n_orig > 0 else 0, len(gen_idx))
        
        if n_orig + n_gen == 0:
            # Skip classe vazia (não append nada)
            continue
        
        # Slices: Agora sempre 4D (graças ao fix em prepare_dataset)
        slice_orig = X_orig[orig_idx] if n_orig > 0 else np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32)
        slice_gen = X_gen[gen_idx[:n_gen]] if n_gen > 0 else np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32)
        
        # Vstack seguro (ambos 4D)
        combined_slice = np.vstack([slice_orig, slice_gen])
        X_mixed.append(combined_slice)
        y_mixed.extend([label] * (n_orig + n_gen))
    
    if not X_mixed:
        # Se todas classes vazias, retorna empty 4D
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32), np.array([], dtype=np.int32)
    
    X_mixed_train = np.vstack(X_mixed)
    y_mixed_train = np.array(y_mixed)
    idx = np.random.permutation(len(y_mixed_train))
    return X_mixed_train[idx], y_mixed_train[idx]

def test_on_data(dataset_dir, model, class_labels):
    X_test, y_test = prepare_dataset(dataset_dir, class_labels)
    y_proba = model.predict(X_test, verbose=0)
    y_pred = y_proba.argmax(axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if len(class_labels) > 2 else roc_auc_score(y_test, y_proba[:, 1])
    metrics = {
        'Accuracy': report['accuracy'],
        'F1': report['weighted avg']['f1-score'],
        'AUROC': auc
    }
    return metrics, y_pred, y_test

def plot_train_history(history_df, title, file_name, cv=1):
    plt.figure(figsize=(8, 6))
    plt.plot(history_df['loss'], label='Train Loss')
    if 'val_loss' in history_df.columns:
        plt.plot(history_df['val_loss'], label='Val Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_confusion_matrix(y_pred, y_test, class_labels, title):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.close()
    return cm
""" 
def deep_custom_cnn(input_shape, num_classes=2):
    return custom_cnn(input_shape, num_classes)
"""

def stratified_split(X, y, train_size=0.7, val_size=0.15, random_state=123):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y, random_state=random_state)
    val_size_adjusted = val_size / (1 - (1-train_size))
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_mixed_dataset(original_dir, generated_dir, class_labels, target_size=(128, 128), mix_ratio=0.5):
    X_orig, y_orig = prepare_dataset(original_dir, class_labels, target_size)
    X_gen, y_gen = prepare_dataset(generated_dir, class_labels, target_size)
    X_mixed = []
    y_mixed = []
    for label in range(len(class_labels)):
        orig_idx = np.where(y_orig == label)[0]
        gen_idx = np.where(y_gen == label)[0]
        n_orig = len(orig_idx)
        n_gen = min(int(n_orig / mix_ratio), len(gen_idx))
        X_mixed.append(np.vstack([X_orig[orig_idx], X_gen[gen_idx[:n_gen]]]))
        y_mixed.extend([label] * (n_orig + n_gen))
    X_mixed_train = np.vstack(X_mixed)
    y_mixed_train = np.array(y_mixed)
    idx = np.random.permutation(len(y_mixed_train))
    return X_mixed_train[idx], y_mixed_train[idx]

# cv_train_and_evaluate_model Atualizado (com suporte a val/test pré-split)
def cv_train_and_evaluate_model(dataset_dir_train, val_dataset_dir=None, test_dataset_dir=None, class_labels=None, model_fn=None, weights=None, input_shape=None, title=None, file_name=None, cv=1, epochs=5, batch_size=32, num_classes=2, use_mix=False, generated_dir=None):
    print(f'Class labels: {class_labels}, Num classes: {num_classes}')
    
    # Carregue dados de treino (mistura se ativada)
    if use_mix and generated_dir:
        X_train_all, y_train_all = prepare_mixed_dataset(dataset_dir_train, generated_dir, class_labels, input_shape[:2])
    else:
        X_train_all, y_train_all = prepare_dataset(dataset_dir_train, class_labels, input_shape[:2])
    
    # Split interno SÓ para train (se não tem val pré-split)
    if val_dataset_dir is None:
        X_temp, X_internal_val, y_temp, y_internal_val = train_test_split(X_train_all, y_train_all, test_size=0.3, stratify=y_train_all, random_state=123)
        X_train, y_train = X_temp, y_temp
        X_val, y_val = X_internal_val, y_internal_val
    else:
        X_train, y_train = X_train_all, y_train_all
        X_val, y_val = prepare_dataset(val_dataset_dir, class_labels, input_shape[:2])
    
    # Treino
    model = model_fn(input_shape, num_classes=num_classes, weights=weights)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val, y_val), verbose=1)
    
    print(f"Debug VGG: len(X_train) = {len(X_train)}, shape = {X_train.shape if len(X_train) > 0 else 'empty'}")
    print(f"Debug VGG: len(y_train) = {len(y_train)}, unique classes = {np.unique(y_train) if len(y_train) > 0 else 'empty'}")
    print(f"Debug VGG: len(X_val) = {len(X_val)}, shape = {X_val.shape if len(X_val) > 0 else 'empty'}")
    if len(X_train) == 0:
       raise ValueError("X_train vazio – verifique prepare_dataset para train.")
    
    # Avaliação
    # Avaliação
    if test_dataset_dir:
        test_metrics, y_pred, y_test = test_on_data(test_dataset_dir, model, class_labels)
        cm = plot_confusion_matrix(y_pred, y_test, class_labels, f'{title} Test')

        # Adiciona cálculo explícito de acurácia se faltar
        if 'Accuracy' not in test_metrics or np.isnan(test_metrics['Accuracy']):
            test_metrics['Accuracy'] = accuracy_score(y_test, y_pred)
        if 'F1' not in test_metrics or np.isnan(test_metrics['F1']):
            test_metrics['F1'] = f1_score(y_test, y_pred, average='weighted')
        if 'AUROC' not in test_metrics or np.isnan(test_metrics['AUROC']):
            try:
                y_proba = model.predict(X_test, verbose=0)
                test_metrics['AUROC'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except Exception:
                test_metrics['AUROC'] = 0.0

    else:
        test_metrics = {'Accuracy': 0.0, 'F1': 0.0, 'AUROC': 0.0}
        y_pred, y_test, cm = None, None, None

    plot_train_history(pd.DataFrame(history.history), title, file_name, cv)
    
    return model, test_metrics, cm


# Stubs para funções missing no import do notebook (baseados em cv_train_and_evaluate_model)
def train_and_evaluate_model(dataset_dir_train, test_dataset_dir, class_labels, model_fn, weights, input_shape, title, file_name, epochs=5, batch_size=32, num_classes=2, use_mix=False, generated_dir=None):
    # Versão sem CV: Chama cv com cv=1
    return cv_train_and_evaluate_model(dataset_dir_train, test_dataset_dir=test_dataset_dir, class_labels=class_labels, model_fn=model_fn, weights=weights, input_shape=input_shape, title=title, file_name=file_name, cv=1, epochs=epochs, batch_size=batch_size, num_classes=num_classes, use_mix=use_mix, generated_dir=generated_dir)

def cv_train_model(X, y, model_fn, epochs=10, batch_size=32, cv=5, num_classes=2):
    # Stub simples: CV com KFold (use seu código original se tiver)
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y_val
        model = model_fn((128, 128, 3), num_classes=num_classes)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
        val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
        fold_metrics.append({'Fold': fold+1, 'Val Acc': val_acc})
    return pd.DataFrame(fold_metrics), model  # Retorna best model (último)
"""
def cv_train_vgg_model(X, y, epochs=10, batch_size=32, cv=5):
    # Stub: CV específico para VGG
    return cv_train_model(X, y, vgg_model, epochs, batch_size, cv, num_classes=2)
"""
def imbalanced_cv_train_and_evaluate_model(dataset_dir_train, test_dataset_dir, class_labels, model_fn, weights, input_shape, title, file_name, cv=1, epochs=5, batch_size=32, num_classes=2, use_mix=False, generated_dir=None):
    # Stub para imbalanced: Usa class_weights (adicione se imbalance)
    from sklearn.utils.class_weight import compute_class_weight
    X, y = prepare_dataset(dataset_dir_train, class_labels) if not use_mix else prepare_mixed_dataset(dataset_dir_train, generated_dir, class_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    # Treino com weights
    model = model_fn(input_shape, num_classes, weights)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict, verbose=1)
    test_metrics, y_pred, y_test = test_on_data(test_dataset_dir, model, class_labels)
    plot_train_history(pd.DataFrame(history.history), title, file_name, cv)
    cm = plot_confusion_matrix(y_pred, y_test, class_labels, f'{title} Test')
    return model, test_metrics, cm

# Outras funções (mantidas)
def swin_transformer_model(input_shape=(224, 224, 3), num_classes=2):
    pass  # Placeholder

def build_densenet_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet', **kwargs):
    base_model = DenseNet121(include_top=False, input_shape=input_shape, weights=weights)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    """
     model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    """
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # NÃO sparse_categorical_crossentropy
    metrics=['accuracy']
    )

    return model


def split_dataset_to_folders(original_dir, output_base, train_ratio=0.7, val_ratio=0.15, random_state=123):
    class_labels = sorted(os.listdir(original_dir))
    for class_label in class_labels:
        class_path = os.path.join(original_dir, class_label)
        if not os.path.exists(class_path):
            continue
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        np.random.seed(random_state)
        np.random.shuffle(files)
        
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        os.makedirs(os.path.join(output_base, 'train', class_label), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'val', class_label), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'test', class_label), exist_ok=True)
        
        for i, file in enumerate(files):
            src = os.path.join(class_path, file)
            if i < n_train:
                dst = os.path.join(output_base, 'train', class_label, file)
            elif i < n_train + n_val:
                dst = os.path.join(output_base, 'val', class_label, file)
            else:
                dst = os.path.join(output_base, 'test', class_label, file)
            shutil.copy(src, dst)  # Copia
    print("Split concluído!")

def analyze_performance(metrics_runs, output_dir='.', save_plot=True):
    if not metrics_runs:
        raise ValueError("metrics_runs não pode estar vazio!")
    
    df_metrics = pd.DataFrame(metrics_runs)
    
    if 'F1' in df_metrics.columns:
        mean_f1 = df_metrics['F1'].mean()
        std_f1 = df_metrics['F1'].std()
        print(f"F1 Mean ± Std (Overall): {mean_f1:.4f} ± {std_f1:.4f}")
    else:
        print("Coluna 'F1' não encontrada; pule análise.")
        return df_metrics, None
    
    if 'Config' in df_metrics.columns:
        configs = df_metrics['Config'].unique()
        comparison_data = []
        for config in configs:
            subset = df_metrics[df_metrics['Config'] == config]
            f1_mean = subset['F1'].mean()
            f1_std = subset['F1'].std()
            auroc_mean = subset['AUROC'].mean() if 'AUROC' in subset.columns else np.nan
            comparison_data.append({
                'Config': config,
                'F1_Mean': round(f1_mean, 4),
                'F1_Std': round(f1_std, 4),
                'AUROC_Mean': round(auroc_mean, 4) if not np.isnan(auroc_mean) else None
            })
        
        comparison = pd.DataFrame(comparison_data)
        print("\nTabela Comparativa:")
        print(comparison)
        
        csv_path = os.path.join(output_dir, 'performance_comparison.csv')
        comparison.to_csv(csv_path, index=False)
        print(f"Tabela salva em: {csv_path}")
    else:
        print("Adicione 'Config' aos dicts para comparação por config.")
        comparison = None
    
    
    if save_plot and 'Config' in df_metrics.columns and 'F1' in df_metrics.columns:
        plt.figure(figsize=(8, 6))
        df_metrics.boxplot(column='F1', by='Config', ax=plt.gca())
        plt.title('F1 Score por Configuração')
        plt.suptitle('')
        plt.xlabel('Configuração')
        plt.ylabel('F1 Score')
        plot_path = os.path.join(output_dir, 'performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot salvo em: {plot_path}")
    elif save_plot:
        print("Não é possível plotar sem 'Config' e 'F1'.")
    
    return df_metrics, comparison


