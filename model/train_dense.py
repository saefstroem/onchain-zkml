import struct
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

FRAC_BITS = 8
I16 = np.iinfo(np.int16)
MAGIC = b"ZNN1"


def build_model(features):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(features,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def q_weights(w):
    return np.clip(np.round(w * (1 << FRAC_BITS)), I16.min, I16.max).astype(np.int64)


def q_bias(b):
    return np.round(b * (1 << (2 * FRAC_BITS))).astype(np.int64)


def quantize(model):
    layers = []
    for layer in model.layers:
        w, b = layer.get_weights()
        layers.append((q_weights(w), q_bias(b), layer.activation is keras.activations.relu))
    return layers


def write_net(path, layers):
    blob = bytearray(MAGIC) + struct.pack("<H", len(layers))
    for w, b, relu in layers:
        blob += struct.pack("<HHB", w.shape[0], w.shape[1], int(relu))
        blob += w.T.astype("<i2").tobytes() + b.astype("<i4").tobytes()
    open(path, "wb").write(blob)


def write_norm(path, scaler, y_scaler):
    mean = np.asarray(scaler.mean_).reshape(-1)
    std = np.asarray(scaler.scale_).reshape(-1)
    blob = struct.pack("<H", mean.shape[0])
    blob += mean.astype("<f4").tobytes() + std.astype("<f4").tobytes()
    blob += struct.pack("<ff", float(y_scaler.mean_[0]), float(y_scaler.scale_[0]))
    open(path, "wb").write(blob)


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.california_housing.load_data(
        test_split=0.2, seed=0
    )
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")

    y_scaler = StandardScaler()
    y_scaler.fit(y_train.reshape(-1, 1))
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).astype("float32").reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).astype("float32").reshape(-1)

    model = build_model(x_train.shape[1])
    model.fit(x_train, y_train, epochs=40, batch_size=64, validation_split=0.1, verbose=0)
    _, mae = model.evaluate(x_test, y_test, verbose=0)
    print(f"test MAE {mae * float(y_scaler.scale_[0]):,.0f} USD over {len(x_test)} homes")

    write_net("engine/src/net.bin", quantize(model))
    write_norm("norm.bin", scaler, y_scaler)
    print(f"wrote engine/src/net.bin, norm.bin ({x_train.shape[1]}->{16}->1)")


if __name__ == "__main__":
    main()
