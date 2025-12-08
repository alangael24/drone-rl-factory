// Physics Code
void physics_step(DroneState* state, float* actions) {
    // Convertir acciones normalizadas
    float thrust = (actions[0] + 1.0f) * 0.5f * 20.0f;
    float roll_cmd = actions[1] * 5.0f;
    float pitch_cmd = actions[2] * 5.0f;
    float yaw_cmd = actions[3] * 5.0f;

    // Actualizar velocidades angulares
    float damping = 0.95f;
    state->roll_rate = state->roll_rate * damping + roll_cmd * 0.1f;
    state->pitch_rate = state->pitch_rate * damping + pitch_cmd * 0.1f;
    state->yaw_rate = state->yaw_rate * damping + yaw_cmd * 0.1f;

    // Integrar orientación
    state->roll += state->roll_rate * DT;
    state->pitch += state->pitch_rate * DT;
    state->yaw += state->yaw_rate * DT;

    // Limitar orientación
    if (state->roll > 1.0f) state->roll = 1.0f;
    if (state->roll < -1.0f) state->roll = -1.0f;
    if (state->pitch > 1.0f) state->pitch = 1.0f;
    if (state->pitch < -1.0f) state->pitch = -1.0f;

    // Calcular fuerzas en frame mundo
    float cos_roll = cosf(state->roll);
    float sin_roll = sinf(state->roll);
    float cos_pitch = cosf(state->pitch);
    float sin_pitch = sinf(state->pitch);

    float thrust_x = thrust * sin_pitch;
    float thrust_y = -thrust * sin_roll * cos_pitch;
    float thrust_z = thrust * cos_roll * cos_pitch;

    // Aceleración
    float mass = state->mass > 0.0f ? state->mass : 1.0f;
    float gravity = state->gravity > 0.0f ? state->gravity : 9.81f;

    float ax = (thrust_x + state->wind_x) / mass;
    float ay = (thrust_y + state->wind_y) / mass;
    float az = ((thrust_z + state->wind_z) / mass) - gravity;

    // Arrastre
    float drag = state->drag_coeff > 0.0f ? state->drag_coeff : 0.1f;
    ax -= state->vx * drag;
    ay -= state->vy * drag;
    az -= state->vz * drag;

    // Integrar velocidad y posición
    state->vx += ax * DT;
    state->vy += ay * DT;
    state->vz += az * DT;

    state->x += state->vx * DT;
    state->y += state->vy * DT;
    state->z += state->vz * DT;

    // Colisión con suelo
    if (state->z < 0.0f) {
        state->z = 0.0f;
        state->vz = 0.0f;
        state->collisions++;
    }

    // Límites
    if (state->z > WORLD_SIZE_Z) state->z = WORLD_SIZE_Z;
    if (state->x > WORLD_SIZE_X) state->x = WORLD_SIZE_X;
    if (state->x < -WORLD_SIZE_X) state->x = -WORLD_SIZE_X;
    if (state->y > WORLD_SIZE_Y) state->y = WORLD_SIZE_Y;
    if (state->y < -WORLD_SIZE_Y) state->y = -WORLD_SIZE_Y;

    state->steps++;
}

// Reward Code
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