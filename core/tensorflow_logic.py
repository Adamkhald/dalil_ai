import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_iris
import numpy as np

class TrainingCallback(Callback):
    def __init__(self, update_func):
        super().__init__()
        self.update_func = update_func

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, acc={logs.get('accuracy', 0):.4f}"
        self.update_func(msg)

class TensorFlowPipeline:
    def __init__(self):
        self.train_ds = None
        self.val_ds = None
        self.model = None
        self.classes = []
        self.img_size = (224, 224)
        self.mode = "IMAGE" # IMAGE, TEXT, TABULAR
        self.vectorizer = None # For text
        
    # --- IMAGE ---
    def load_image_data(self, data_dir, batch_size=32, val_split=0.2):
        self.mode = "IMAGE"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
            
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=batch_size
        )
        
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=batch_size
        )
        
        self.classes = self.train_ds.class_names
        
        # Optimize
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return f"Loaded data. Classes: {self.classes}"

    def load_tabular_data(self, X, y):
        self.mode = "TABULAR"
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
        self.train_ds = dataset
        self.val_ds = dataset # Simplicity
        return "Loaded tabular data."
        
    def load_tabular_class_data(self, X, y, num_classes):
        self.mode = "TABULAR_CLASS"
        import numpy as np
        y = tf.keras.utils.to_categorical(y, num_classes)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
        self.train_ds = dataset
        self.val_ds = dataset
        self.classes = [str(i) for i in range(num_classes)] # Placeholder
        return "Loaded Tabular Classification data."
        
    def load_text_data(self, texts, labels):
        self.mode = "TEXT"
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        dataset = dataset.shuffle(1000).batch(32)
        self.train_ds = dataset.take(int(len(texts)/32 * 0.8))
        self.val_ds = dataset.skip(int(len(texts)/32 * 0.8))
        
        self.vectorizer = layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
        self.vectorizer.adapt(dataset.map(lambda text, label: text))
        return "Loaded Text data."

    def load_timeseries_data(self, X, y):
        self.mode = "TIMESERIES"
        # X shape: (samples, time_steps, features)
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)
        self.train_ds = dataset
        self.val_ds = dataset
        return "Loaded Time Series data."

    def load_builtin_data(self, name):
        import numpy as np
        if name == "MNIST":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
            x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
            # Resize for MobileNet if needed or keep simpler models
            # For simplicity, we stick to Custom CNN for MNIST or resize
            self.mode = "IMAGE_BUILTIN"
            self.classes = [str(i) for i in range(10)]
            self.img_size = (28, 28)
            # Create TF Dataset
            self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
            self.val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
            return "Loaded MNIST (Image Class)."
            
        elif name == "Fashion MNIST":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
            x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
            self.mode = "IMAGE_BUILTIN"
            self.classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
            self.img_size = (28, 28)
            self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
            self.val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
            return "Loaded Fashion MNIST (Image Class)."
            
        elif name == "CIFAR-10":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            self.mode = "IMAGE_BUILTIN"
            self.classes = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
            self.img_size = (32, 32)
            self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
            self.val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
            return "Loaded CIFAR-10 (Image Class)."

        elif name == "IMDB Reviews":
             self.mode = "TEXT"
             # Use top 10000 words
             vocab_size = 10000
             (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
             
             # Padding for simple batching
             x_train = tf.keras.utils.pad_sequences(x_train, maxlen=200)
             x_test = tf.keras.utils.pad_sequences(x_test, maxlen=200)
             
             # Create dataset
             self.train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
             self.val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
             
             # Override vectorizer since IMDB is already integer-encoded
             # We just need an embedding that accepts ints
             self.vectorizer = layers.Lambda(lambda x: x) # No-op
             return "Loaded IMDB Reviews (Text Class)."

        elif name == "California Housing":
             self.mode = "TABULAR"
             data = fetch_california_housing()
             X = data.data.astype("float32")
             y = data.target.astype("float32")
             
             # Split manually
             split = int(len(X) * 0.8)
             self.train_ds = tf.data.Dataset.from_tensor_slices((X[:split], y[:split])).shuffle(1000).batch(32)
             self.val_ds = tf.data.Dataset.from_tensor_slices((X[split:], y[split:])).batch(32)
             return "Loaded California Housing (Tabular Reg)."

        elif name == "Iris Plants":
             self.mode = "TABULAR_CLASS"
             data = load_iris()
             X = data.data.astype("float32")
             y = data.target.astype("int32")
             self.classes = list(data.target_names)
             
             y_cat = tf.keras.utils.to_categorical(y, 3)
             
             # Shuffle
             idx = np.random.permutation(len(X))
             X, y_cat = X[idx], y_cat[idx]
             
             split = int(len(X) * 0.8)
             self.train_ds = tf.data.Dataset.from_tensor_slices((X[:split], y_cat[:split])).batch(16)
             self.val_ds = tf.data.Dataset.from_tensor_slices((X[split:], y_cat[split:])).batch(16)
             return "Loaded Iris Plants (Tabular Class)."

        elif name == "Sine Wave (Synthetic)":
             self.mode = "TIMESERIES"
             t = np.linspace(0, 100, 2000)
             data = np.sin(t) + np.random.normal(0, 0.1, 2000)
             
             # Create sequences
             # Input: 10 steps, Output: Next step
             X_seq, y_seq = [], []
             for i in range(len(data)-11):
                 X_seq.append(data[i:i+10])
                 y_seq.append(data[i+10])
             
             X_seq = np.array(X_seq).astype("float32")
             y_seq = np.array(y_seq).astype("float32")
             X_seq = np.expand_dims(X_seq, -1) # (N, 10, 1)
             
             split = int(len(X_seq) * 0.8)
             self.train_ds = tf.data.Dataset.from_tensor_slices((X_seq[:split], y_seq[:split])).batch(32)
             self.val_ds = tf.data.Dataset.from_tensor_slices((X_seq[split:], y_seq[split:])).batch(32)
             return "Loaded Synthetic Sine Wave (TimeSeries)."
            
        return "Unknown dataset."

    def build_model(self, model_name="mobilenet_v2"):
        if self.mode == "TABULAR":
            self.model = models.Sequential([layers.Dense(64, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(1)])
            return "Built Tabular Regressor."
            
        elif self.mode == "TABULAR_CLASS":
             self.model = models.Sequential([
                layers.Dense(64, activation='relu'), 
                layers.Dense(32, activation='relu'), 
                layers.Dense(len(self.classes), activation='softmax')
             ])
             return "Built Tabular Classifier."
             
        elif self.mode == "TEXT":
            self.model = models.Sequential([
                self.vectorizer,
                layers.Embedding(10001, 64, mask_zero=True),
                layers.Bidirectional(layers.LSTM(64)),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='sigmoid') # Binary for now
            ])
            return "Built LSTM Text Classifier."
            
        elif self.mode == "TIMESERIES":
             self.model = models.Sequential([
                layers.LSTM(64, return_sequences=False),
                layers.Dense(32, activation='relu'),
                layers.Dense(1)
             ])
             return "Built TimeSeries LSTM."
        
        # Image logic fallback
        return self._build_image_model(model_name)

    def _build_image_model(self, model_name):
        if not self.classes:
            raise ValueError("Load data first to determine classes.")
            
        num_classes = len(self.classes)
        
        if model_name == "mobilenet_v2":
            base_model = applications.MobileNetV2(input_shape=self.img_size + (3,), include_top=False, weights='imagenet')
        elif model_name == "resnet50":
            base_model = applications.ResNet50(input_shape=self.img_size + (3,), include_top=False, weights='imagenet')
        else:
             raise ValueError("Unknown Image Model or Custom CNN removed.")

        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        # Preprocessing internal to model for deployment
        if model_name == "resnet50":
            x = applications.resnet50.preprocess_input(inputs)
        else: # mobilenet
            x = applications.mobilenet_v2.preprocess_input(inputs)
            
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes)(x) # No activation, using from_logits=True
        
        self.model = tf.keras.Model(inputs, outputs)
        
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
        return f"Built {model_name} with Transfer Learning."

    def train(self, epochs=5, optimizer_name="adam", learning_rate=0.001, update_callback=None):
        if self.model is None or self.train_ds is None:
            raise ValueError("Model or Data not ready.")
            
        # Re-compile with chosen settings
        if optimizer_name == "adam":
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = "adam"
            
        # Determine loss
        # Heuristic: if Tabular -> MSE, else CrossEntropy
        if self.mode in ["TABULAR", "TIMESERIES"]:
            loss = "mse"
            metrics = ["mae"]
        elif self.mode == "TABULAR_CLASS":
             loss = "categorical_crossentropy"
             metrics = ["accuracy"]
        elif self.mode == "TEXT":
             loss = "binary_crossentropy"
             metrics = ["accuracy"]
        else: # IMAGE
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = ["accuracy"]
            
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
            
        callbacks = []
        if update_callback:
            callbacks.append(TrainingCallback(update_callback))
            
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        self.history_data = history.history
        return "Training Finished."

    def plot_results(self):
        if not hasattr(self, 'history_data'):
            return None
        
        hist = self.history_data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Loss Plot
        ax1.plot(hist['loss'], label='Train Loss')
        if 'val_loss' in hist:
            ax1.plot(hist['val_loss'], label='Val Loss')
        ax1.set_title(f'Loss ({self.mode})')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Metric Plot
        metric = "accuracy" if "accuracy" in hist else "mae"
        if metric in hist:
            ax2.plot(hist[metric], label=f'Train {metric.capitalize()}')
            if f'val_{metric}' in hist:
                ax2.plot(hist[f'val_{metric}'], label=f'Val {metric.capitalize()}')
            ax2.set_title(f'{metric.capitalize()}')
            ax2.set_xlabel('Epoch')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No Metric Available', ha='center')
            
        plt.tight_layout()
        return fig

    def visualize_predictions(self):
        """Generates a plot for TimeSeries or Regression evaluation."""
        if self.model is None or self.val_ds is None:
            return None
            
        fig = plt.figure(figsize=(10, 5))
        
        if self.mode == "TIMESERIES":
            # Taking one batch
            for x_batch, y_batch in self.val_ds.take(1):
                preds = self.model.predict(x_batch)
                
                # Plot first 3 samples
                for i in range(min(3, len(y_batch))):
                    plt.subplot(1, 3, i+1)
                    # Input sequence is x_batch[i] (10, 1)
                    seq = x_batch[i].numpy().flatten()
                    true_next = y_batch[i].numpy()
                    pred_next = preds[i][0]
                    
                    plt.plot(range(len(seq)), seq, 'o-', label='History')
                    plt.plot(len(seq), true_next, 'gx', markersize=10, label='True Next')
                    plt.plot(len(seq), pred_next, 'rx', markersize=10, label='Pred Next')
                    plt.legend(fontsize='small')
                    plt.title(f'Sample {i+1}')
                    
            plt.tight_layout()
            return fig
            
        elif self.mode == "TABULAR": # Regression
            # Plot Actual vs Predicted for a batch
            y_true = []
            y_pred = []
            for x, y in self.val_ds.take(5): # Take a few batches
                p = self.model.predict(x)
                y_true.extend(y.numpy().flatten())
                y_pred.extend(p.flatten())
            
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            return fig
            
        return None

    def predict_single(self, input_data):
        if self.model is None:
            return "Error: Train model first."
            
        try:
            if self.mode == "TEXT":
                # input_data is string
                pred = self.model.predict([input_data])
                score = pred[0][0]
                return f"Sentiment score: {score:.4f} ({'Positive' if score>0.5 else 'Negative'})"
                
            elif self.mode == "TABULAR_CLASS":
                 # input_data is comma-sep string of numbers
                 vec = [float(x) for x in input_data.split(",")]
                 vec = np.array([vec])
                 pred = self.model.predict(vec)
                 cls = np.argmax(pred)
                 return f"Predicted Class: {self.classes[cls]} (Conf: {pred[0][cls]:.2f})"
                 
            elif self.mode == "TABULAR":
                 vec = [float(x) for x in input_data.split(",")]
                 vec = np.array([vec])
                 pred = self.model.predict(vec)
                 return f"Predicted Value: {pred[0][0]:.4f}"
                 
            elif "IMAGE" in self.mode:
                # input_data is path
                img = tf.keras.utils.load_img(input_data, target_size=self.img_size)
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                if self.mode == "IMAGE":
                     # MobileNet expects specific preprocessing
                     img_array = applications.mobilenet_v2.preprocess_input(img_array)
                else:
                     img_array = img_array / 255.0
                     
                preds = self.model.predict(img_array)
                print(preds) # Debug
                if hasattr(self, 'classes') and self.classes:
                    score = tf.nn.softmax(preds[0]) # Assuming logits
                    cls = self.classes[np.argmax(score)]
                    return f"Class: {cls} ({100*np.max(score):.2f}%)"
                else:
                    return str(preds)

        except Exception as e:
            return f"Prediction Error: {e}"
        
        return "Not implemented for this mode."

    def export_tflite(self, path="model.tflite"):
        if self.model is None:
            raise ValueError("No model to export.")
            
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        with open(path, 'wb') as f:
            f.write(tflite_model)
            
        return f"Saved TFLite model to {path}"
