#include "led.h"
#include "driver/ledc.h"

void init_led() {
    // main led
    gpio_set_direction(LEDC_GPIO_NUM, GPIO_MODE_OUTPUT);
    gpio_set_level(LEDC_GPIO_NUM, 0);

    // debug led
    gpio_set_direction(LEDD_GPIO_NUM, GPIO_MODE_OUTPUT);
    gpio_set_level(LEDD_GPIO_NUM, 1);

    ledc_timer_config_t ledc_timer = {
        .duty_resolution = LEDC_TIMER_13_BIT,
        .freq_hz = 5000,
        .speed_mode = LEDC_HIGH_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&ledc_timer);

    ledc_channel_config_t ledc_channel = {
        .channel = LEDC_CHANNEL_0,
        .duty = 0,
        .gpio_num = LEDC_GPIO_NUM,
        .intr_type = LEDC_INTR_DISABLE,
        .speed_mode = LEDC_HIGH_SPEED_MODE,
        .timer_sel = LEDC_TIMER_0,
    };
    ledc_channel_config(&ledc_channel);
    ledc_set_fade_with_time(LEDC_HIGH_SPEED_MODE, LEDC_CHANNEL_0, 0, 0);
}

void set_led_duty(int value) {
    assert(value >= 0 && value <= (1 << 13) - 1);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, LEDC_CHANNEL_0, value);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, LEDC_CHANNEL_0);
}

void set_debug_led(int value) {
    assert(value == 0 || value == 1);
    gpio_set_level(LEDD_GPIO_NUM, value);
}