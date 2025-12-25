import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import Callback

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
        
    def load_data(self, data_dir, batch_size=32, val_split=0.2):
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

    def build_model(self, model_name="mobilenet_v2"):
        if not self.classes:
            raise ValueError("Load data first to determine classes.")
            
        num_classes = len(self.classes)
        
        if model_name == "mobilenet_v2":
            base_model = applications.MobileNetV2(input_shape=self.img_size + (3,), include_top=False, weights='imagenet')
        elif model_name == "resnet50":
            base_model = applications.ResNet50(input_shape=self.img_size + (3,), include_top=False, weights='imagenet')
        else:
            # Simple CNN fallback
            self.model = models.Sequential([
                layers.Rescaling(1./255, input_shape=self.img_size + (3,)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes)
            ])
            self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
            return "Built Custom CNN."

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

    def train(self, epochs=5, update_callback=None):
        if self.model is None or self.train_ds is None:
            raise ValueError("Model or Data not ready.")
            
        callbacks = []
        if update_callback:
            callbacks.append(TrainingCallback(update_callback))
            
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        return "Training Finished."

    def export_tflite(self, path="model.tflite"):
        if self.model is None:
            raise ValueError("No model to export.")
            
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        with open(path, 'wb') as f:
            f.write(tflite_model)
            
        return f"Saved TFLite model to {path}"
