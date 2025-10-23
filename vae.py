import tensorflow as tf
from tensorflow.keras import layers, Model

latent_dim = 2
input_dim = 16  # example feature size

# ----- Encoder -----
encoder_inputs = tf.keras.Input(shape=(input_dim,))
x = layers.Dense(32, activation="relu")(encoder_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# ----- Decoder -----
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(latent_inputs)
decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")

# ----- VAE Model -----
outputs = decoder(encoder(encoder_inputs)[2])
vae = Model(encoder_inputs, outputs, name="vae")

# ----- Custom Loss (Inside a Layer-Safe Context) -----
# Use Keras backend or tf ops *inside add_loss*, not standalone tf.fn calls on symbolic tensors
z_mean, z_log_var, z = encoder(encoder_inputs)
reconstruction_loss = tf.reduce_mean(
    tf.keras.losses.mse(encoder_inputs, outputs)
)

kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
)

vae.add_loss(reconstruction_loss + kl_loss)
vae.add_metric(reconstruction_loss, name="reconstruction_loss")
vae.add_metric(kl_loss, name="kl_loss")

vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.summary()