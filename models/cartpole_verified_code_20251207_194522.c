// Physics Code
void physics_step(CartPoleState* state, float* actions) {
    // Parámetros físicos
    float force = actions[0] * 10.0f;  // Convertir de [-1,1] a [-10,10]

    float cart_mass = state->cart_mass > 0.0f ? state->cart_mass : 1.0f;
    float pole_mass = state->pole_mass > 0.0f ? state->pole_mass : 0.1f;
    float pole_length = state->pole_length > 0.0f ? state->pole_length : 0.5f;

    float total_mass = cart_mass + pole_mass;
    float pole_half = pole_length * 0.5f;

    float sin_theta = sinf(state->pole_angle);
    float cos_theta = cosf(state->pole_angle);

    // Ecuaciones del péndulo invertido
    float temp = (force + pole_mass * pole_half * state->pole_velocity * state->pole_velocity * sin_theta) / total_mass;

    float theta_acc = (DEFAULT_GRAVITY * sin_theta - cos_theta * temp) /
                      (pole_half * (4.0f/3.0f - pole_mass * cos_theta * cos_theta / total_mass));

    float x_acc = temp - pole_mass * pole_half * theta_acc * cos_theta / total_mass;

    // Integración de Euler
    state->cart_velocity += x_acc * DT;
    state->cart_position += state->cart_velocity * DT;

    state->pole_velocity += theta_acc * DT;
    state->pole_angle += state->pole_velocity * DT;

    state->steps++;
}

// Reward Code
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