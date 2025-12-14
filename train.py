import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from data import load_kmnist, set_seed
from models import build_mlp, build_cnn
from viz import plot_curves, plot_confusion_matrix


def evaluate_model(model, test_ds):
    y_true = []
    y_pred = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'cnn'], default='cnn')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    train_ds, val_ds, test_ds, class_names = load_kmnist(
        batch_size=args.batch_size,
        seed=args.seed
    )

    if args.model == 'mlp':
        model = build_mlp(num_classes=len(class_names))
    else:
        model = build_cnn(num_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    ckpt_path = f'models/{args.model}_seed{args.seed}.h5'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    if args.evaluate:
        print('🔄 Carregando pesos...')
        model.load_weights(ckpt_path)
    else:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks
        )
        plot_curves(
            history,
            outpath=f'figures/curvas_{args.model}_seed{args.seed}.png'
        )

    acc, f1, y_true, y_pred = evaluate_model(model, test_ds)

    print(f'✅ Teste accuracy: {acc:.4f}')
    print(f'✅ Teste F1-macro: {f1:.4f}')

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        outpath=f'figures/cm_{args.model}_seed{args.seed}.png'
    )


if __name__ == '__main__':
    main()
