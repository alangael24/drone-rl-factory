float calculate_reward(RoboticArm2DState* state) {
    // Distancia del end effector al objetivo
    float dx = state->target_x - state->end_effector_x;
    float dy = state->target_y - state->end_effector_y;
    float distance = sqrtf(dx*dx + dy*dy);

    // Recompensa por cercanía
    float reward = 1.0f / (1.0f + distance * 2.0f);

    // Bonus por alcanzar objetivo
    if (distance < 0.1f) {
        reward += 10.0f;
    } else if (distance < 0.3f) {
        reward += 3.0f;
    }

    // Penalización por velocidad angular alta (eficiencia energética)
    float vel_penalty = (fabsf(state->joint1_velocity) + fabsf(state->joint2_velocity)) * 0.1f;
    reward -= vel_penalty;

    return reward;
}