class CoppeliaSimEnvWrapper(EnvironmentSpec):
    ...
    def step(self, action, feedback=None):
        done = False
        info = {}
        prev_distance_to_goal = self.distance_to_goal()

        # Apply feedback if provided
        if feedback is not None:
            action += feedback

        # Make a step in simulation
        self.apply_controls(action)
        self.env.step()
        self.step_counter += 1

        # Reward calculations
        success_reward = self.success_check()
        distance_reward = (prev_distance_to_goal - self.distance_to_goal()) / self.initial_distance

        reward = distance_reward + success_reward

        # Check reset conditions
        if self.step_counter > self.max_step_count:
            done = True
            logging.info('--------Reset: Timeout--------')
        elif self.distance_to_goal() > 0.8:
            done = True
            logging.info('--------Reset: Too far from target--------')
        elif self.collision_check():
            done = True
            logging.info('--------Reset: Collision--------')

        return self.get_observation(), reward, done, info
    ...