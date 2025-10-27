import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Sampling Layer ---
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Encoder ---
latent_dim = 2

encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(64, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# --- Decoder ---
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(64, activation='relu')(latent_inputs)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
decoder_outputs = layers.Reshape((28, 28, 1))(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")

# --- VAE ---
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Store for loss calculation
        self.kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return reconstructed

# --- Custom loss wrapper ---
def vae_loss_fn(x, x_reconstructed, kl_loss):
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(x, x_reconstructed)
    )
    total_loss = reconstruction_loss + kl_loss
    return total_loss

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss=lambda x, x_rec: vae_loss_fn(x, x_rec, vae.kl_loss))

# --- Train on dummy data (MNIST-like) ---
import numpy as np
x_train = np.random.rand(1000, 28, 28, 1).astype("float32")
vae.fit(x_train, x_train, epochs=3, batch_size=32)