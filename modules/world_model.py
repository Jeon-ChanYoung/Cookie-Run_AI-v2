import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal, Independent
from .network import Encoder, Decoder, RecurrentModel, TransitionModel, RepresentationModel

class WorldModel:
    def __init__(self, config):
        self.config = config

        self.encoder = Encoder(self.config).to(self.config.device)
        self.decoder = Decoder(self.config).to(self.config.device)
        self.recurrent_model = RecurrentModel(self.config).to(self.config.device)
        self.transition_model = TransitionModel(self.config).to(self.config.device)
        self.representation_model = RepresentationModel(self.config).to(self.config.device)

        self.world_model_parameters = (
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.recurrent_model.parameters()) +
            list(self.transition_model.parameters()) +
            list(self.representation_model.parameters())
        )

        self.world_model_optimizer = optim.AdamW(
            self.world_model_parameters,
            lr=self.config.world_model_lr,
            weight_decay=self.config.world_model_weight_decay
        )

        self.load_checkpoint(self.config.model_path)
        print("✅ WorldModel parameters loaded from checkpoint.")

    def dynamic_learning(self, experiences):
        states, actions = experiences

        encoded_states = self.encoder(states)

        hidden = torch.zeros(self.config.batch_size, self.config.recurrent_size, device=self.config.device)
        latent = torch.zeros(self.config.batch_size, self.config.latent_size, device=self.config.device)

        hiddens = torch.zeros(self.config.batch_size, self.config.batch_length-1, self.config.recurrent_size, device=self.config.device)
        latents = torch.zeros(self.config.batch_size, self.config.batch_length-1, self.config.latent_size, device=self.config.device)
        priors_logits = torch.zeros(self.config.batch_size, self.config.batch_length-1, self.config.latent_length, self.config.latent_classes, device=self.config.device)
        posteriors_logits = torch.zeros(self.config.batch_size, self.config.batch_length-1, self.config.latent_length, self.config.latent_classes, device=self.config.device)

        for t in range(1, self.config.batch_length):
            hidden = self.recurrent_model(hidden, latent, actions[:, t-1])
            _, prior_logits = self.transition_model(hidden)
            latent, posterior_logits = self.representation_model(hidden, encoded_states[:, t])

            hiddens[:, t-1] = hidden
            latents[:, t-1] = latent
            priors_logits[:, t-1] = prior_logits
            posteriors_logits[:, t-1] = posterior_logits


        ############# compute loss #############

        # Reconstruction loss
        reconstruction_means = self.decoder(hiddens, latents)
        reconstruction_dist = Independent(
            Normal(reconstruction_means, 1),
            len(self.config.observation_shape)
        )
        reconstruction_loss = -reconstruction_dist.log_prob(states[:, 1:]).mean()

        # KL loss
        prior_loss = self.compute_kl(posteriors_logits.detach(), priors_logits)
        prior_loss = self.config.prior_coefficient * prior_loss.clamp(min=self.config.free_nat)
        posterior_loss = self.compute_kl(posteriors_logits, priors_logits.detach())
        posterior_loss = self.config.posterior_coefficient * posterior_loss.clamp(min=self.config.free_nat)
        kl_loss = (prior_loss + posterior_loss).mean()

        loss = reconstruction_loss + kl_loss

        ############# backprop #############

        self.world_model_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.world_model_parameters, 
            self.config.gradient_clip, 
            self.config.gradient_norm_type
        )
        self.world_model_optimizer.step()

        return reconstruction_loss.item(), kl_loss.item()

    def compute_kl(self, logits_p, logits_q):
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        return (p * (log_p - log_q)).sum(dim=(-2, -1))

    def save_model_params(self, episode, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{self.config.map_name}_ep{episode}.pth')
        
        torch.save({
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrent_model'       : self.recurrent_model.state_dict(),
            'transition_model'      : self.transition_model.state_dict(),
            'representation_model'  : self.representation_model.state_dict(),
            'world_model_optimizer' : self.world_model_optimizer.state_dict()
        }, save_path)
        
        print(f"💾 Model saved: {save_path}")

    def load_checkpoint(self, check_point_path):
        print(f"📁 Loading checkpoint: {check_point_path}")
        checkpoint = torch.load(check_point_path, map_location=self.config.device)

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrent_model.load_state_dict(checkpoint['recurrent_model'])
        self.transition_model.load_state_dict(checkpoint['transition_model'])
        self.representation_model.load_state_dict(checkpoint['representation_model'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        print("Checkpoint loaded successfully.")