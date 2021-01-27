# wange011.github.io
```python
"""
Parameters
__________
- smooth_reg_steps    : Number of optimization steps for adversarial perturbation
- lr_reg              : Learning rate for the optimization step described above
- smooth_reg_epsilon  : Radius of the ball
- gamma_reg           : Coefficient for smoooth regularization within the policy loss
"""

batch_size, _ = state_batch.view(-1, self.obs_dim).size()

# Intitializing the delta of the perturbation
delta = torch.rand(batch_size, self.obs_dim, requires_grad=True) * smooth_reg_epsilon
delta = delta.to(self.device)

# Initialize within the l2 ball
# while(torch.norm(delta, 2) > smooth_reg_epsilon):
    # delta = torch.rand(batch_size, obs_size) * smooth_reg_epsilon

dist_state = self.get_action_dist(state_batch.view(-1, self.obs_dim))

# Optimize the smoothness regularizer
for i in range(smooth_reg_steps):
    # Perturbed state
    state_bar = state_batch.view(-1, self.obs_dim) + delta
    
    dist_state_bar = self.get_action_dist(state_bar)
    
    # Calculate Jeffrey’s divergence of the action distribution for the original state and the perturbed state
    div = 1/2 * torch.distributions.kl_divergence(dist_state, dist_state_bar) + 1/2 * torch.distributions.kl_divergence(dist_state_bar, dist_state)
    
    # Ascend the gradient to find adversarial perturbation
    delta += lr_reg * grad(outputs=div, inputs=delta, grad_outputs=torch.ones_like(div))[0]

    # Project onto l-infinity ball
    delta = torch.clamp(delta, -smooth_reg_epsilon, smooth_reg_epsilon)
    
    # To project onto the l-2 ball, the delta is of most of length epsilon
    # if torch.norm(delta, 2) > smooth_reg_epsilon:
        # delta /= torch.norm(delta, 2)
        # delta *= smooth_reg_epsilon

# Calculate the regularization term
state_bar = state_batch.view(-1, self.obs_dim) + delta
dist_state_bar = self.get_action_dist(state_bar)
reg = (1/2 * torch.distributions.kl_divergence(dist_state, dist_state_bar)
    + 1/2 * torch.distributions.kl_divergence(dist_state_bar, dist_state)).mean()
policy_loss += gamma_reg * reg
```
