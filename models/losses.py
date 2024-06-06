import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=128):
        super(MelSpectrogramLoss, self).__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                  n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  n_mels=n_mels)

    def forward(self, audio_seq, mel_spec_orig):
        mel_spec_transform = self.mel_transform(audio_seq)
        print(mel_spec_orig.shape, mel_spec_transform.shape)
        loss = F.mse_loss(mel_spec_orig, mel_spec_transform)
        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self, discriminator):
        super(FeatureMatchingLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, real, fake):
        real_features = self.discriminator.extract_features(real)
        fake_features = self.discriminator.extract_features(fake)
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.mse_loss(real_feat, fake_feat)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self, discriminator):
        super(GeneratorLoss, self).__init__()
        self.adversarial_loss = AdversarialLoss()
        self.mel_loss = MelSpectrogramLoss()
        self.feature_matching_loss = FeatureMatchingLoss(discriminator)
        self.lambda_fm = 2
        self.lambda_mel = 45

    def forward(self, real_audio, fake_audio, mel_spec_orig, disc_fake):
        adv_loss = self.adversarial_loss(disc_fake, torch.ones_like(disc_fake))
        mel_loss = self.mel_loss(fake_audio, mel_spec_orig)
        fm_loss = self.feature_matching_loss(real_audio, fake_audio)
        return adv_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.adversarial_loss = AdversarialLoss()

    def forward(self, disc_real, disc_fake):
        real_loss = self.adversarial_loss(disc_real, torch.ones_like(disc_real))
        fake_loss = self.adversarial_loss(disc_fake, torch.zeros_like(disc_fake))
        return real_loss + fake_loss


# Mock discriminator class for feature extraction
class MockDiscriminator(nn.Module):
    def __init__(self):
        super(MockDiscriminator, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def extract_features(self, x):
        return [x]  # Dummy feature extraction


if __name__ == "__main__":
    batch_size = 16
    audio_seq_length = 47998
    generator_output_shape = [batch_size, 1, audio_seq_length]

    mel_channels = 1
    mel_filters = 128
    mel_timeframes = 100
    mel_spectrogram_shape = [batch_size, mel_channels, mel_filters, mel_timeframes]

    msd_disc_shape = [batch_size, 1, 2]
    mcd_disc_shape = [batch_size, 1, 2]

    x_mel = torch.randn(mel_spectrogram_shape)
    x_generated = torch.randn(generator_output_shape)

    labels = torch.sigmoid(torch.randn(msd_disc_shape))
    msd_disc_output = torch.sigmoid(torch.randn(msd_disc_shape))
    mcd_disc_output = torch.sigmoid(torch.randn(mcd_disc_shape))
