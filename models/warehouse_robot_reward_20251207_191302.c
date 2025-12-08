float calculate_reward(WarehouseRobotState* state) {
    // Distancia al objetivo
    float dx = state->target_x - state->x;
    float dy = state->target_y - state->y;
    float distance = sqrtf(dx*dx + dy*dy);

    // Recompensa por cercanía
    float reward = 2.0f - distance * 0.1f;

    // Bonus por llegar
    if (distance < 0.3f) {
        reward += 10.0f;
    }

    // Premiar velocidad moderada (no muy lento, no muy rápido)
    float speed = fabsf(state->v_linear);
    if (speed > 0.3f && speed < 1.5f) {
        reward += 0.5f;
    } else if (speed > 2.0f) {
        reward -= (speed - 2.0f) * 0.5f;
    }

    // Penalizar giros bruscos
    reward -= fabsf(state->v_angular) * 0.2f;

    // Penalizar cercanía a obstáculos
    if (state->obstacle_front < 0.5f) {
        reward -= (0.5f - state->obstacle_front) * 2.0f;
    }

    return reward;
}