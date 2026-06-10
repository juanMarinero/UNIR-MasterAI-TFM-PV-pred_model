#!/usr/bin/env python3

#  vim: set ts=4 sts=4 sw=4 expandtab tw=0 foldcolumn=2 foldmethod=indent :

import random
import numpy as np
import tensorflow as tf

# Set the logging level to only show error messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def test_set_seed_reproducibility():
    # Primera ejecución con seed fijo
    set_seed(42)
    rand1 = random.random()
    np_rand1 = np.random.random()
    tf_rand1 = tf.random.uniform(shape=()).numpy()

    # Segunda ejecución con el mismo seed
    set_seed(42)
    rand2 = random.random()
    np_rand2 = np.random.random()
    tf_rand2 = tf.random.uniform(shape=()).numpy()

    # Verificar que los valores son iguales
    assert rand1 == rand2, f"random: {rand1} != {rand2}"
    assert np_rand1 == np_rand2, f"numpy: {np_rand1} != {np_rand2}"
    assert tf_rand1 == tf_rand2, f"tensorflow: {tf_rand1} != {tf_rand2}"

    print("✓ Prueba de reproducibilidad exitosa!")


def test_different_seeds_produce_different_values():
    # Seed 42
    set_seed(42)
    rand_42 = random.random()

    # Seed 123
    set_seed(123)
    rand_123 = random.random()

    # Verificar que sean diferentes (con alta probabilidad)
    assert rand_42 != rand_123, "Diferentes seeds produjeron el mismo valor"

    print("✓ Prueba de seeds diferentes exitosa!")


def test_sequence_reproducibility():
    # Generar secuencias con el mismo seed
    set_seed(42)
    seq1 = [random.randint(1, 100) for _ in range(10)]

    set_seed(42)
    seq2 = [random.randint(1, 100) for _ in range(10)]

    assert seq1 == seq2, f"Secuencias diferentes: {seq1} vs {seq2}"

    print("✓ Prueba de secuencia reproducible exitosa!")


def test_tensorflow_model_reproducibility():
    set_seed(42)

    # Crear un modelo simple
    model1 = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            tf.keras.layers.Dense(1),
        ]
    )

    # Generar algunos pesos iniciales
    weights1 = [layer.get_weights() for layer in model1.layers]

    # Reiniciar con el mismo seed
    set_seed(42)
    model2 = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
            tf.keras.layers.Dense(1),
        ]
    )
    weights2 = [layer.get_weights() for layer in model2.layers]

    # Comparar pesos (deberían ser iguales)
    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        np.testing.assert_array_almost_equal(w1[0], w2[0], decimal=5)
        if len(w1) > 1 and len(w2) > 1:  # para biases
            np.testing.assert_array_almost_equal(w1[1], w2[1], decimal=5)

    print("✓ Prueba de reproducibilidad de TensorFlow exitosa!")


def main():
    print("Ejecutando pruebas...")
    test_set_seed_reproducibility()
    test_different_seeds_produce_different_values()
    test_sequence_reproducibility()
    test_tensorflow_model_reproducibility()
    print("\n🎉 Todas las pruebas pasaron exitosamente!")


if __name__ == "__main__":
    main()
