float calculate_reward(DroneState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float dz = state->target_z - state->z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);

    // Recompensa inversamente proporcional a la distancia
    float reward = 1.0f / (1.0f + distance);

    // Bonus por estar cerca
    if (distance < 0.2f) {
        reward += 5.0f;
    }

    // Penalización por orientación extrema
    float orientation_penalty = (fabsf(state->roll) + fabsf(state->pitch)) * 0.5f;
    reward -= orientation_penalty;

    // Penalización por velocidad angular alta
    float angular_penalty = (fabsf(state->roll_rate) + fabsf(state->pitch_rate) + fabsf(state->yaw_rate)) * 0.1f;
    reward -= angular_penalty;

    // Penalización por colisión
    if (state->collisions > 0) {
        reward -= 5.0f;
    }

    return reward;
}