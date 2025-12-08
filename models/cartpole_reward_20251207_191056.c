float calculate_reward(CartPoleState* state) {
    // Recompensa por mantener el palo vertical
    float angle_penalty = fabsf(state->pole_angle);

    // Recompensa por mantener el carro centrado
    float position_penalty = fabsf(state->cart_position) * 0.1f;

    // Recompensa base por sobrevivir
    float reward = 1.0f;

    // Penalizaciones
    reward -= angle_penalty * 2.0f;
    reward -= position_penalty;

    // Penalización por velocidades altas
    reward -= fabsf(state->pole_velocity) * 0.1f;
    reward -= fabsf(state->cart_velocity) * 0.05f;

    return reward;
}