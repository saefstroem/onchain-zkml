import struct
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

MAGIC = b"ZNN2"


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


def extract(model):
    layers = []
    for layer in model.layers:
        w, b = layer.get_weights()
        layers.append((w, b, layer.activation is keras.activations.relu))
    return layers


def write_net(path, layers):
    blob = bytearray(MAGIC) + struct.pack("<H", len(layers))
    for w, b, relu in layers:
        blob += struct.pack("<HHB", w.shape[0], w.shape[1], int(relu)) # 2byte2byte 1 byte
        blob += w.T.astype("<f4").tobytes() + b.astype("<f4").tobytes()
    open(path, "wb").write(blob)


def write_norm(path, scaler, y_scaler):
    mean = np.asarray(scaler.mean_).reshape(-1)
    std = np.asarray(scaler.scale_).reshape(-1)
    blob = struct.pack("<H", mean.shape[0]) # 2 byte 
    blob += mean.astype("<f4").tobytes() + std.astype("<f4").tobytes() #f32, serializes whole arr
    blob += struct.pack("<ff", float(y_scaler.mean_[0]), float(y_scaler.scale_[0])) #f32
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

    write_net("engine/src/net.bin", extract(model))
    write_norm("engine/src/norm.bin", scaler, y_scaler)
    print("wrote engine/src/net.bin, engine/src/norm.bin")


if __name__ == "__main__":
    main()
