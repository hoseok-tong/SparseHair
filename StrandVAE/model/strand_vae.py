import torch
import torch.nn as nn
import torch.nn.functional as F

from StrandVAE.model.component.layers import *
from StrandVAE.util.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
EPSILON = 1e-7



class StrandVAE(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=256, dim_out=3, num_layers=5, w0_initial=30., latent_dim=64, coord_length=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.layer1 = nn.Conv1d(coord_length, latent_dim*2, 3)
        self.layer2 = nn.Conv1d(latent_dim*2, latent_dim*2, 1)
        self.layer3 = nn.Conv1d(latent_dim*2, latent_dim*4, 1)

        self.layer41 = nn.Linear(latent_dim*4, latent_dim)  # mu
        self.layer42 = nn.Linear(latent_dim*4, latent_dim)  # logvar

        self.dec = ModulatedSiren(dim_in, dim_hidden, dim_out, num_layers, w0_initial, latent_dim, coord_length)
    
    def encode(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x).view(-1, self.latent_dim*4))
        return self.layer41(x), self.layer42(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        gen = self.dec(z)
        
        return gen, mu, logvar, z

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        gen = self.dec(z)
        return gen
    

class StrandVAE_1(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=256, dim_out=3, num_layers=5, w0_initial=30., latent_dim=64, coord_length=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.layer0 = nn.Linear(1, 1)

        self.layer1 = nn.Conv1d(coord_length, latent_dim*2, 3)
        self.layer2 = nn.Conv1d(latent_dim*2, latent_dim*2, 1)
        self.layer3 = nn.Conv1d(latent_dim*2, latent_dim*4, 1)

        self.layer41 = nn.Linear(latent_dim*4, latent_dim)  # mu
        self.layer42 = nn.Linear(latent_dim*4, latent_dim)  # logvar

        self.dec = ModulatedSiren(dim_in, dim_hidden, dim_out, num_layers, w0_initial, latent_dim, coord_length)
    
    def encode(self, x):
        totalLen, directions = self.pos2dir(x)
        total_length_vector = totalLen.unsqueeze(2).repeat(1, 1, 3)  # (batch_size, 1, 3)
        encoder_input = torch.cat([total_length_vector, directions], dim=1)  # (batch_size, 100, 3)        

        x = self.layer1(encoder_input)
        x = self.layer2(x)
        x = self.layer3(x).view(-1, self.latent_dim*4)
        return self.layer41(x), self.layer42(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        gen = self.dec(z)
        return gen, mu, logvar, z

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        gen = self.dec(z)
        return gen
    
    @staticmethod
    def pos2dir(p):
        directions = p[:, 1:, :] - p[:, :-1, :]  # (batch_size, 99, 3)
        segment_lengths = torch.norm(directions, dim=2)  # (batch_size, 99)
        total_length = torch.sum(segment_lengths, dim=1, keepdim=True)  # (batch_size, 1)
        return total_length, directions
        


class StrandVAE_2(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=256, dim_out=6, num_layers=5, w0_initial=30., latent_dim=64, coord_length=99):
        super().__init__()
        self.latent_dim = latent_dim

        self.layer0 = nn.Linear(1, 1)

        self.layer1 = nn.Conv1d(coord_length, latent_dim*2, 7)
        self.layer2 = nn.Conv1d(latent_dim*2, latent_dim*2, 1)
        self.layer3 = nn.Conv1d(latent_dim*2, latent_dim*4, 1)

        self.layer41 = nn.Linear(latent_dim*4, latent_dim)  # mu
        self.layer42 = nn.Linear(latent_dim*4, latent_dim)  # logvar

        self.dec = ModulatedSiren(dim_in, dim_hidden, dim_out, num_layers, w0_initial, latent_dim, coord_length)
    
    def encode(self, x):
        rotations, length = self.cartesian_to_rotational_repr(x)  # (batch_size, 99, 6), (batch_size, 99)
        length_vector = length.unsqueeze(2)  # (batch_size, 99, 1)
        encoder_input = torch.cat([length_vector, rotations], dim=2)  # (batch_size, 99, 7)

        x = self.layer1(encoder_input)
        x = self.layer2(x)
        x = self.layer3(x).view(-1, self.latent_dim*4)
        return self.layer41(x), self.layer42(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        gen = self.dec(z)
        return gen, mu, logvar, z

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        gen = self.dec(z)
        return gen
    
    def cartesian_to_rotational_repr(self, position: torch.Tensor):
        direction = F.normalize(position[..., 1:, :] - position[..., :-1, :], dim=-1)   # unit vector
        forward = torch.zeros_like(direction)
        forward[..., 1] = -1
        rotation = self.rotation_between_vectors(forward, direction)  # (..., num_samples - 1, 3, 3)

        rotation = matrix_to_rotation_6d(rotation)  # (..., num_samples - 1, 6)

        length = torch.norm(position[..., 1:, :] - position[..., :-1, :], dim=-1)  # (..., num_samples - 1)

        return rotation, length
    
    def rotation_between_vectors(self, v1: torch.Tensor, v2: torch.Tensor):
        """ Compute the rotation matrix `R` between unit vectors `v1` and `v2`, such that `v2 = Rv1`.
        Args:
            v1/v2 (torch.Tensor): 3D unit vectors of shape (..., 3).
        Returns:
            (torch.Tensor): Rotation matrices of shape (..., 3, 3).
        """
        axis = torch.cross(v1, v2, dim=-1)
        axis = F.normalize(axis, dim=-1)
        angle = self.dot(v1, v2).clamp(min=-1.0, max=1.0)
        # resolve singularity when angle=pi, since both angle 0 and angle pi will produce zero axes
        v_clone = v1[angle == -1]
        axis_ortho = torch.zeros_like(v_clone)
        axis_ortho[..., 0] = v_clone[..., 2] - v_clone[..., 1]
        axis_ortho[..., 1] = v_clone[..., 0] - v_clone[..., 2]
        axis_ortho[..., 2] = v_clone[..., 1] - v_clone[..., 0]
        # if two vectors v1 and v2 point in opposite directions (angle=pi), modify the axis to be orthogonal to v1
        axis[angle == -1] = F.normalize(axis_ortho, dim=-1)
        # angle = acos_linear_extrapolation(angle)  # [0, pi]
        angle = torch.acos(angle.clamp(min=-1.0 + EPSILON, max=1.0 - EPSILON))  # [0, pi]

        axis_angle = axis * angle[..., None]
        rotmat = axis_angle_to_matrix(axis_angle)

        return rotmat    
    
    def dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return torch.einsum('...n,...n->...', x, y)



class StrandVAE_3(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=256, dim_out=6, num_layers=5, w0_initial=30., latent_dim=64, coord_length=99):
        super().__init__()
        self.latent_dim = latent_dim

        self.embed_class = nn.Linear(1, 99)

        self.layer1 = nn.Conv1d(coord_length, latent_dim*2, 7)
        self.layer2 = nn.Conv1d(latent_dim*2, latent_dim*2, 1)
        self.layer3 = nn.Conv1d(latent_dim*2, latent_dim*4, 1)
        self.leaky_relu = nn.LeakyReLU()

        self.layer41 = nn.Linear(latent_dim*4, latent_dim)  # mu
        self.layer42 = nn.Linear(latent_dim*4, latent_dim)  # logvar

        self.decoder_input = nn.Linear(latent_dim+1, latent_dim)
        self.dec = ModulatedSiren(dim_in, dim_hidden, dim_out, num_layers, w0_initial, latent_dim, coord_length)
    
    def encode(self, x):
        rotations, length = self.cartesian_to_rotational_repr(x)  # (batch_size, 99, 6), (batch_size, 99)
        self.length_label = length.sum(dim=1).unsqueeze(1)    # (batch_size, 1)
        embedded_class = (self.embed_class(self.length_label)).unsqueeze(2)     # (batch_size, 99, 1)
        encoder_input = torch.cat([rotations, embedded_class], dim=2)  # (batch_size, 99, 7)

        x = self.layer1(encoder_input)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.leaky_relu(x)
        x = self.layer3(x)
        x = self.leaky_relu(x)
        x = x.view(-1, self.latent_dim*4)
        return self.layer41(x), self.layer42(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) # (batch_size, 64)
        z = torch.cat([z, self.length_label], dim=1)    # (batch_size, 65)
        z = self.decoder_input(z)    # (batch_size, 64)
        gen = self.dec(z)
        return gen, mu, logvar, z

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        gen = self.dec(z)
        return gen
    
    def cartesian_to_rotational_repr(self, position: torch.Tensor):
        direction = F.normalize(position[..., 1:, :] - position[..., :-1, :], dim=-1)   # unit vector
        forward = torch.zeros_like(direction)
        forward[..., 1] = -1
        rotation = self.rotation_between_vectors(forward, direction)  # (..., num_samples - 1, 3, 3)

        rotation = matrix_to_rotation_6d(rotation)  # (..., num_samples - 1, 6)

        length = torch.norm(position[..., 1:, :] - position[..., :-1, :], dim=-1)  # (..., num_samples - 1)

        return rotation, length
    
    def rotation_between_vectors(self, v1: torch.Tensor, v2: torch.Tensor):
        """ Compute the rotation matrix `R` between unit vectors `v1` and `v2`, such that `v2 = Rv1`.
        Args:
            v1/v2 (torch.Tensor): 3D unit vectors of shape (..., 3).
        Returns:
            (torch.Tensor): Rotation matrices of shape (..., 3, 3).
        """
        axis = torch.cross(v1, v2, dim=-1)
        axis = F.normalize(axis, dim=-1)
        angle = self.dot(v1, v2).clamp(min=-1.0, max=1.0)
        # resolve singularity when angle=pi, since both angle 0 and angle pi will produce zero axes
        v_clone = v1[angle == -1]
        axis_ortho = torch.zeros_like(v_clone)
        axis_ortho[..., 0] = v_clone[..., 2] - v_clone[..., 1]
        axis_ortho[..., 1] = v_clone[..., 0] - v_clone[..., 2]
        axis_ortho[..., 2] = v_clone[..., 1] - v_clone[..., 0]
        # if two vectors v1 and v2 point in opposite directions (angle=pi), modify the axis to be orthogonal to v1
        axis[angle == -1] = F.normalize(axis_ortho, dim=-1)
        # angle = acos_linear_extrapolation(angle)  # [0, pi]
        angle = torch.acos(angle.clamp(min=-1.0 + EPSILON, max=1.0 - EPSILON))  # [0, pi]

        axis_angle = axis * angle[..., None]
        rotmat = axis_angle_to_matrix(axis_angle)

        return rotmat
    
    def dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return torch.einsum('...n,...n->...', x, y)


