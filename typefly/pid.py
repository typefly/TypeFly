import math

class PID:
    def __init__(self, kp, ki, kd, rate, cutoff_freq=0.0, i_limit=0.0, o_limit=0.0):
        self.init(kp, ki, kd, rate, cutoff_freq, i_limit, o_limit)

    def init(self, kp, ki, kd, rate, cutoff_freq=0.0, i_limit=0.0, o_limit=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.rate = rate
        self.dt = 1.0 / rate
        self.i_limit = i_limit
        self.o_limit = o_limit
        self.last_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0
        self.tau = 1.0 / (2.0 * math.pi * cutoff_freq) if cutoff_freq > 0.0 else 0.0

    def update(self, error):
        output = self.kp * error

        raw_derivative = (error - self.last_error) * self.rate
        self.last_error = error

        if self.tau > 0.0:
            alpha = self.dt / (self.tau + self.dt)
            self.filtered_derivative = (
                self.filtered_derivative * (1.0 - alpha) + raw_derivative * alpha
            )
        else:
            self.filtered_derivative = raw_derivative

        output += self.kd * self.filtered_derivative

        self.integral += error * self.dt
        if self.i_limit > 0.0:
            self.integral = max(-self.i_limit, min(self.integral, self.i_limit))

        output += self.ki * self.integral

        if self.o_limit > 0.0:
            output = max(-self.o_limit, min(output, self.o_limit))

        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.filtered_derivative = 0.0