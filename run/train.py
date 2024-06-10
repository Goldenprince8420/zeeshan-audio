import torch
import torch.nn as nn

from models.models import Generator, Discriminator
from models.losses import GeneratorLoss, DiscriminatorLoss


def do_train(dataloader,
             run_config,
             data_config,
             model_config,
             output_config,
             do_save=False):
    epochs = run_config["epochs"]
    batch_size = data_config["batch_size"]
    audio_length = data_config["audio_length"]

    rand_tensor = torch.randn([batch_size, 1, audio_length])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    params_generator = model_config["generator_params"]
    params_discriminator = model_config["discriminator_params"]

    generator = Generator(params=params_generator)

    discriminator = Discriminator(params=params_discriminator)
    discriminator(rand_tensor)

    generator_loss = GeneratorLoss(discriminator=discriminator)
    discriminator_loss = DiscriminatorLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=run_config["generator_learning_rate"],
                                   betas=(0.5, 0.999))

    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=run_config["discriminator_learning_rate"],
                                   betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_audio,
                meta,
                labels,
                mel_spec,
                mel_spec_db) in enumerate(dataloader):
            # Move data to device
            real_audio = real_audio
            real_audio = real_audio.transpose(1, 2)

            x_mel = torch.unsqueeze(mel_spec, dim=1)
            x_meta = meta.transpose(1, 2).to(torch.float32)

            generated_audio = generator(x_mel, x_meta)
            # print("Generated Audio Shape: ", generated_audio.shape)

            # Concatenate generated and real audio
            combined_audio = torch.cat((generated_audio.detach(), real_audio.detach()), dim=0)

            # Pass through the discriminator
            combined_discriminator_output_msd, \
            combined_discriminator_output_mcd, \
            combined_discriminator_output_msd_features, \
            combined_discriminator_output_mcd_features = discriminator(combined_audio)

            # Split the outputs
            batch_size = real_audio.size(0)
            discriminator_output_msd_generated = combined_discriminator_output_msd[:batch_size]
            discriminator_output_msd_real = combined_discriminator_output_msd[batch_size:]

            discriminator_output_mcd_generated = combined_discriminator_output_mcd[:batch_size]
            discriminator_output_mcd_real = combined_discriminator_output_mcd[batch_size:]

            discriminator_output_msd_initiator_features_combined = combined_discriminator_output_msd_features[0]
            discriminator_output_msd_distributor_features_combined = combined_discriminator_output_msd_features[1]
            discriminator_output_mcd_initiator_features_combined = combined_discriminator_output_mcd_features[0]
            discriminator_output_mcd_convolver_features_combined = combined_discriminator_output_mcd_features[1]

            discriminator_output_msd_initiator_features_generated = discriminator_output_msd_initiator_features_combined[:batch_size]
            discriminator_output_msd_initiator_features_real = discriminator_output_msd_initiator_features_combined[batch_size:]

            discriminator_output_msd_distributor_features_generated = discriminator_output_msd_distributor_features_combined[:batch_size]
            discriminator_output_msd_distributor_features_real = discriminator_output_msd_distributor_features_combined[batch_size:]

            discriminator_output_mcd_initiator_features_generated = discriminator_output_mcd_initiator_features_combined[:batch_size]
            discriminator_output_mcd_initiator_features_real = discriminator_output_mcd_initiator_features_combined[batch_size:]

            discriminator_output_mcd_convolver_features_generated = discriminator_output_mcd_convolver_features_combined[:batch_size]
            discriminator_output_mcd_convolver_features_real = discriminator_output_mcd_convolver_features_combined[batch_size:]

            # Train the Generator
            optimizer_G.zero_grad()
            # Calculate the generator loss
            gen_loss = generator_loss(real_audio,
                                      generated_audio,
                                      discriminator_output_msd_generated,
                                      discriminator_output_mcd_generated,
                                      discriminator_output_msd_initiator_features_generated,
                                      discriminator_output_msd_distributor_features_generated,
                                      discriminator_output_mcd_initiator_features_generated,
                                      discriminator_output_mcd_convolver_features_generated,
                                      discriminator_output_msd_initiator_features_real,
                                      discriminator_output_msd_distributor_features_real,
                                      discriminator_output_mcd_initiator_features_real,
                                      discriminator_output_mcd_convolver_features_real)

            # BackProp
            gen_loss.backward(retain_graph=True)
            # Update Optimizer
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Calculate the discriminator loss
            disc_loss = discriminator_loss(discriminator_output_msd_generated,
                                           discriminator_output_msd_real,
                                           discriminator_output_mcd_generated,
                                           discriminator_output_mcd_real)
            disc_loss.backward()
            optimizer_D.step()

            break
        break


if __name__ == "__main__":
    # Data Preparation
    data_params = {
        "batch_size": 2,
        "num_channels": 1,
        "mel_filters": 128,
        "timeframes": 100,
        "audio_length": 47998
    }

    batch_size = data_params["batch_size"]
    num_channels = data_params["num_channels"]
    mel_filters = data_params["mel_filters"]
    timeframes = data_params["timeframes"]
    audio_length = data_params["audio_length"]
    shape = (batch_size, num_channels, mel_filters, timeframes)

    # Create random input tensor with the shape (batch_size, num_channels, mel_filters, timeframes)
    x_mel = torch.randn(batch_size, num_channels, mel_filters, timeframes)
    print("Mel Initiator Input Shape: ", x_mel.shape)

    params = {
        "generator_params": {
            "mel_initiator_params": {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "dpn_depth": 3,
                "audio_length": 47998,
                "input_shape": shape
            },
            "meta_initiator_params": {
                "input_size": 10,
                "hidden_size": 64,
                "output_size": 32
            },
        },
        "discriminator_params": {
            "msd_params": {
                "params_avgpool": {
                    "kernel_size": 11,
                    "stride": 4
                },
                "params_distributor": {
                    "defconv_in_channels": 1,
                    "defconv_out_channels": 16,
                    "kernel_size": 3,
                    "kernel_list": [3, 5],
                    "stride": 1,
                    "depth": 4
                },
                "params_final_msd": {
                    "kernel_size": 7,
                    "stride": 4,
                    "hidden_dims": 512
                }
            },
            "mcd_params": {
                "initiator_params": {
                    "height": 103,
                    "width": 466
                },
                "convolver_params": {
                    "in_channels": 1,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "kernel_list": [3, 5],
                    "stride": 1,
                    "depth": 3
                },
                "final_params": {
                    "in_features": 0,  # To be modified later
                    "hidden_features": 512,
                    "out_features": 2
                }
            }
        }

    }

    # Generator Params
    params_generator = params["generator_params"]

    # Discriminator Params
    discriminator_params = params["discriminator_params"]

    # Define model and input parameters
    input_size = 10  # Input size

    # Generate random input
    x_meta = torch.randn(batch_size, input_size)
    print("Meta Initiator Input Shape: ", x_meta.shape)

    generator = Generator(params=params_generator)

    out_gen = generator(x_mel, x_meta)
    print("Generator Output Shape: ", out_gen.shape)

    # Inputs
    generated_samples = torch.randn(batch_size, 1, audio_length)
    real_samples = torch.randn(batch_size, 1, audio_length)

    # Labels
    # labels_generated = torch.zeros(batch_size, dtype=torch.long)  # Label 0
    # labels_real = torch.ones(batch_size, dtype=torch.long)  # Label 1

    discriminator_model = Discriminator(params=discriminator_params)

    discriminator_output_msd_generated, discriminator_output_mcd_generated, \
    discriminator_params_msd_generated, discriminator_params_mcd_generated = discriminator_model(generated_samples)
    discriminator_output_msd_real, discriminator_output_mcd_real = discriminator_output_msd_generated, discriminator_output_mcd_generated  # discriminator_model(generated_samples)
    print("Discriminator MSD Output Shape: ", discriminator_output_msd_generated.shape)
    print("Discriminator MCD Output Shape: ", discriminator_output_mcd_generated.shape)
#
#     # generator_loss = GeneratorLoss(discriminator=discriminator_model)
#     # discriminator_loss = DiscriminatorLoss()
#     #
#     # # Calculate the generator loss
#     # gen_loss = generator_loss(real_samples, generated_samples, x_mel, discriminator_output_msd_generated)
#     # print(f"Generator Loss: {gen_loss.item()}")
#     #
#     # # Calculate the discriminator loss
#     # disc_loss = discriminator_loss(discriminator_output_msd_real, discriminator_output_msd_generated)
#     # print(f"Discriminator Loss: {disc_loss.item()}")
